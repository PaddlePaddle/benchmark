// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <paddle/fluid/inference/paddle_inference_api.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>

DEFINE_string(model_dir, "", "model directory");
DEFINE_string(data, "", "input data path");
DEFINE_int32(repeat, 1, "repeat");
DEFINE_bool(print_outputs, false, "Whether to output the prediction results.");
DEFINE_bool(use_gpu, false, "Whether to output the prediction results.");
DEFINE_bool(use_analysis, false, "Whether to output the prediction results.");


template <typename T>
void GetValueFromStream(std::stringstream *ss, T *t) {
  (*ss) >> (*t);
}

template <>
void GetValueFromStream<std::string>(std::stringstream *ss, std::string *t) {
  *t = ss->str();
}

// Split string to vector
template <typename T>
void Split(const std::string &line, char sep, std::vector<T> *v) {
  std::stringstream ss;
  T t;
  for (auto c : line) {
    if (c != sep) {
      ss << c;
    } else {
      GetValueFromStream<T>(&ss, &t);
      v->push_back(std::move(t));
      ss.str({});
      ss.clear();
    }
  }

  if (!ss.str().empty()) {
    GetValueFromStream<T>(&ss, &t);
    v->push_back(std::move(t));
    ss.str({});
    ss.clear();
  }
}

template <typename T>
constexpr paddle::PaddleDType GetPaddleDType();

template <>
constexpr paddle::PaddleDType GetPaddleDType<int64_t>() {
  return paddle::PaddleDType::INT64;
}

template <>
constexpr paddle::PaddleDType GetPaddleDType<float>() {
  return paddle::PaddleDType::FLOAT32;
}

// Parse tensor from string
template <typename T>
bool ParseTensor(const std::string &field, paddle::PaddleTensor *tensor) {
  std::vector<std::string> data;
  Split(field, ':', &data);
  if (data.size() < 2) return false;

  std::string shape_str = data[0];

  std::vector<int> shape;
  Split(shape_str, ' ', &shape);

  std::string mat_str = data[1];

  std::vector<T> mat;
  Split(mat_str, ' ', &mat);

  tensor->shape = shape;
  auto size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) *
      sizeof(T);
  tensor->data.Resize(size);
  std::copy(mat.begin(), mat.end(), static_cast<T *>(tensor->data.data()));
  tensor->dtype = GetPaddleDType<T>();

  return true;
}

// Parse input tensors from string
bool ParseLine(const std::string &line,
               std::vector<paddle::PaddleTensor> *tensors) {
  std::vector<std::string> fields;
  Split(line, ';', &fields);

  // if (fields.size() != 7) return false;

  tensors->clear();
  tensors->reserve(4);

  int i = 0;
  // src_ids
  paddle::PaddleTensor src_ids;
  ParseTensor<int64_t>(fields[i++], &src_ids);
  src_ids.name = "placeholder_0";
  tensors->push_back(src_ids);

  // pos_ids
  paddle::PaddleTensor pos_ids;
  ParseTensor<int64_t>(fields[i++], &pos_ids);
  pos_ids.name = "placeholder_1";
  tensors->push_back(pos_ids);

  // sent_ids
  paddle::PaddleTensor sent_ids;
  ParseTensor<int64_t>(fields[i++], &sent_ids);
  sent_ids.name = "placeholder_2";
  tensors->push_back(sent_ids);

  // input_mask
  paddle::PaddleTensor input_mask;
  ParseTensor<float>(fields[i++], &input_mask);
  input_mask.name = "placeholder_3";
  tensors->push_back(input_mask);

  return true;
}

bool LoadInputData(std::vector<std::vector<paddle::PaddleTensor>> *inputs) {
  if (FLAGS_data.empty()) {
    LOG(ERROR) << "please set input data path";
    return false;
  }

  std::ifstream fin(FLAGS_data);
  std::string line;

  int lineno = 0;
  while (std::getline(fin, line)) {
    std::vector<paddle::PaddleTensor> feed_data;
    if (!ParseLine(line, &feed_data)) {
      LOG(ERROR) << "Parse line[" << lineno << "] error!";
    } else {
      inputs->push_back(std::move(feed_data));
    }
    lineno++;
  }

  LOG(INFO) << "Load " << lineno << " samples from " << FLAGS_data;
  return true;
}

void PrintOutputs(const std::vector<paddle::PaddleTensor> &outputs, int id) {
  //LOG(INFO) << "example_id\tcontradiction\tentailment\tneutral";
  for (size_t i = 0; i < outputs.front().data.length() / sizeof(float) / 3; i += 1) {
    std::cout.flags(std::ios::right);
    std::cout << "example " << std::setw(5) << id << ", "
              << std::setw(12) << static_cast<float *>(outputs[0].data.data())[3 * i] << ", "
              << std::setw(12) << static_cast<float *>(outputs[0].data.data())[3 * i + 1] << ", "
              << std::setw(12) << static_cast<float *>(outputs[0].data.data())[3 * i + 2] << std::endl;
  }
}

template <typename ConfigType>
void SetConfig(ConfigType* config, std::string model_dir, bool use_gpu, bool use_zerocopy) {
  config->model_dir = model_dir;
  if (use_gpu) {
    config->use_gpu = true;
    config->device = 0;
    config->fraction_of_gpu_memory = 0.15;
  } else {
    config->use_gpu = false;
    config->SetCpuMathLibraryNumThreads(10);
  }
}

template <>
void SetConfig<paddle::AnalysisConfig>(paddle::AnalysisConfig* config, std::string model_dir,
                               bool use_gpu, bool use_zerocopy) {
  config->SetModel(model_dir);
  if (use_gpu) {
    config->EnableUseGpu(100, 0);
  } else {
    config->DisableGpu();
    config->EnableMKLDNN();
    config->SetCpuMathLibraryNumThreads(10);
  }
  config->SwitchIrOptim(true);
  // config.SwitchSpecifyInputNames();
  if (use_zerocopy) {
    config->SwitchUseFeedFetchOps(false);
  }
  // config->SwitchIrDebug();
}

std::unique_ptr<paddle::PaddlePredictor> CreatePredictor(
        const paddle::PaddlePredictor::Config *config, bool use_analysis = true) {
  const auto *analysis_config = reinterpret_cast<const paddle::AnalysisConfig *>(config);
  if (use_analysis) {
    return paddle::CreatePaddlePredictor<paddle::AnalysisConfig>(*analysis_config);
  }
  auto native_config = analysis_config->ToNativeConfig();
  return paddle::CreatePaddlePredictor<paddle::NativeConfig>(native_config);
}

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(*argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_model_dir.empty()) {
    LOG(ERROR) << "please set model dir";
    return -1;
  }

  paddle::AnalysisConfig config;
  SetConfig<paddle::AnalysisConfig>(&config, FLAGS_model_dir, FLAGS_use_gpu, false);
  auto predictor = CreatePredictor(
      reinterpret_cast<paddle::PaddlePredictor::Config *>(&config), FLAGS_use_analysis);

  std::vector<std::vector<paddle::PaddleTensor>> inputs;
  if (!LoadInputData(&inputs)) {
    LOG(ERROR) << "load input data error!";
    return -1;
  }

  std::vector<paddle::PaddleTensor> fetch;
  int total_time{0};
  int num_samples{0};
  for (int i = 0; i < FLAGS_repeat; i++) {
    int id = 0;
    for (auto feed : inputs) {
      fetch.clear();

      auto start = std::chrono::system_clock::now();
      predictor->Run(feed, &fetch);
      auto end = std::chrono::system_clock::now();

      if (FLAGS_print_outputs && i == 0) {
        PrintOutputs(fetch, id++);
      }
      if (!fetch.empty()) {
        total_time +=
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                .count();
        //num_samples += fetch.front().data.length() / 2 / sizeof(float);
        num_samples += fetch.front().data.length() / (sizeof(float) * 3);
      }
    }
  }

  auto per_sample_ms =
      static_cast<float>(total_time) / num_samples;
  LOG(INFO) << "Run " << num_samples
            << " samples, average latency: " << per_sample_ms
            << "ms per sample.";

  return 0;
}
