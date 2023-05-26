# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.pipelines.latent_diffusion.pipeline_latent_diffusion import LDMBertModel
from transformers import AutoTokenizer


class LatentDiffusionModel(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_args.pretrained_model_name_or_path,
            model_max_length=model_args.model_max_length,
            subfolder="tokenizer",
        )

        self.vae = AutoencoderKL.from_pretrained(model_args.pretrained_model_name_or_path, subfolder="vqvae")
        self.vae.requires_grad_(False)

        self.text_encoder = LDMBertModel.from_pretrained(
            model_args.pretrained_model_name_or_path, subfolder="bert", use_cache=False
        )
        self.unet = UNet2DConditionModel.from_pretrained(model_args.pretrained_model_name_or_path, subfolder="unet")

        assert model_args.prediction_type in ["epsilon", "v_prediction"]
        self.prediction_type = model_args.prediction_type
        self.noise_scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            prediction_type=self.prediction_type,
        )
        self.register_buffer("alphas_cumprod", self.noise_scheduler.alphas_cumprod)

        # make sure unet text_encoder in train mode, vae in eval mode
        self.unet.train()
        self.text_encoder.train()
        self.vae.eval()

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def get_velocity(self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity

    def forward(self, input_ids=None, pixel_values=None, **kwargs):
        with torch.no_grad():
            self.vae.eval()
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * 0.18215
            noise = torch.randn(latents.shape, device=latents.device)
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                dtype=torch.long,
                device=latents.device,
            )
            noisy_latents = self.add_noise(latents, noise, timesteps)

        encoder_hidden_states = self.text_encoder(input_ids)[0]
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Get the target for loss depending on the prediction type
        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "v_prediction":
            target = self.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.prediction_type}")

        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none").mean([1, 2, 3]).mean()

        return loss
