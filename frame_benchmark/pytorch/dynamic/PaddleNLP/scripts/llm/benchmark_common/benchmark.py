import os
from dataclasses import dataclass, field
from typing import Optional
import logging
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    LlamaTokenizer,
    TrainingArguments,
    Trainer
)

logger = logging.getLogger(__name__)


def get_lora_target_modules(model_name_or_path):
    # Not yet support RowParallelLinear
    if "THUDM/chatglm-6b" in model_name_or_path:
        target_modules = [
            ".*query_key_value.*", 
            ".*dense.*", 
            ".*dense_h_to_4h.*", 
            ".*dense_4h_to_h.*"]
    elif "THUDM/chatglm2-6b" in model_name_or_path:
        target_modules = [
            ".*query.*",
            ".*key.*",
            ".*value.*",
            ".*dense.*",
            ".*dense_h_to_4h.*",
            ".*dense_4h_to_h.*",
        ]
    elif "bloom" in model_name_or_path:
        target_modules = [
            ".*query_key_value.*", 
            ".*dense.*", 
            ".*dense_h_to_4h.*", 
            ".*dense_4h_to_h.*"
        ]
    elif "llama" in model_name_or_path:
        target_modules = [
            ".*q_proj.*",
            ".*v_proj.*",
            ".*k_proj.*",
            ".*o_proj.*",
            ".*gate_proj.*",
            ".*down_proj.*",
            ".*up_proj.*",
        ]
    else:
        raise ValueError(f"Unknown model name: {model_name_or_path}.")
    return target_modules
    

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    lora: Optional[bool] = field(default=False, metadata={"help": "whether to use LoRA"})
    model_name_or_path: str = field(default=None, metadata={"help": "model name or local path"})

@dataclass
class DataArguments:
    src_length: int = field(default=1024, metadata={"help": "The maximum length of source(context) tokens."})
    max_length: int = field(
        default=2048,
        metadata={
            "help": "The maximum length that model input tokens can have. When intokens is set to True, it's also the maximum length for InTokens data stream"
        },
    )
    dataset_name_or_path: str = field(default=None, metadata={"help": "Name or path for dataset"})

def main():
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    model_args, training_args, data_args = parser.parse_args_into_dataclasses()

    if "llama" in model_args.model_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=False)
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    torch_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
    if "chatglm" in model_args.model_name_or_path:
        # Add empty_init=False for zero3 training, refer to https://github.com/THUDM/ChatGLM-6B/issues/530
        model = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            empty_init=False if training_args.deepspeed is not None else True,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch_dtype,
        )
    if model_args.lora:
        target_modules = get_lora_target_modules(model_args.model_name_or_path)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, target_modules=target_modules, r=8, lora_alpha=16, lora_dropout=0.0
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    if model_args.lora and training_args.gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    def preprocess_function(example, src_length, max_length):
        if "src" in example:
            inputs =  example["src"]
            targets = example["tgt"]
        else:
            inputs = example["instruction"]
            if "input" in example:
                inputs += example["input"]
            targets = example["output"]

        model_inputs = tokenizer(
            inputs, 
            max_length=src_length, 
            truncation=True, 
            return_attention_mask=False)

        max_tgt_length = max_length - len(model_inputs["input_ids"])
        labels = tokenizer(targets, max_length=max_tgt_length, truncation=True, return_attention_mask=False)["input_ids"]
        if len(labels) < max_tgt_length:
            labels += [tokenizer.eos_token_id]
        model_inputs["labels"] = [-100] * len(model_inputs["input_ids"]) + labels
        model_inputs["input_ids"] = model_inputs["input_ids"] + labels

        return model_inputs

    dataset = load_dataset("json", data_files={"train":os.path.join(data_args.dataset_name_or_path, "train.json")})["train"]
    dataset = dataset.map(lambda example: preprocess_function(example, data_args.src_length, data_args.max_length), remove_columns=["src", "tgt"])
    data_collator = DataCollatorForSeq2Seq(return_tensors="pt", tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        data_collator=data_collator,
        callbacks=[]
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    train_metrics = trainer.train()

    total_effective_tokens = sum([len(i["input_ids"]) for i in trainer.train_dataset]) * training_args.num_train_epochs
    effective_tokens_per_second = total_effective_tokens / train_metrics.metrics["train_runtime"]
    train_loss = train_metrics.metrics["train_loss"]
    print(f"train_loss: {train_loss:.2f} ")
    print(f"Effective_Tokens_per_second: {effective_tokens_per_second:.2f} ")


if __name__ == "__main__":
    main()
