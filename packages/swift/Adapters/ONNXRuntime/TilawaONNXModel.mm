// SPDX-License-Identifier: MIT
#import "TilawaONNXModel.h"
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <memory>
#include <vector>
#include <string>

struct ModelState {
  Ort::Env environment{ORT_LOGGING_LEVEL_WARNING, "TilawaSwift"};
  Ort::SessionOptions options;
  std::unique_ptr<Ort::Session> session;
  Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<std::string> names;
  std::vector<std::string> outputs;
  std::vector<std::vector<int64_t>> shapes;
  std::vector<bool> integers;
  std::vector<Ort::Value> states;
  int64_t window = 45, mels = 80;

  void reset() {
    states.clear();
    for (size_t i = 0; i < shapes.size(); ++i) {
      auto count = size_t(1);
      for (auto d : shapes[i]) count *= d;
      Ort::AllocatorWithDefaultOptions allocator;
      auto tensor = Ort::Value::CreateTensor(allocator, shapes[i].data(), shapes[i].size(),
        integers[i] ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
      if (integers[i]) std::fill_n(tensor.GetTensorMutableData<int64_t>(), count, int64_t(0));
      else std::fill_n(tensor.GetTensorMutableData<float>(), count, 0.0f);
      states.push_back(std::move(tensor));
    }
  }
};

@implementation TilawaONNXModel {
  std::unique_ptr<ModelState> _model;
}

static void assignError(NSError **error, const std::exception &e) {
  if (error) *error = [NSError errorWithDomain:@"TilawaInference" code:1
    userInfo:@{NSLocalizedDescriptionKey: [NSString stringWithUTF8String:e.what()]}];
}

- (instancetype)initWithPath:(NSString *)path manifest:(NSData *)data error:(NSError **)error {
  self = [super init];
  if (!self) return nil;
  NSDictionary *manifest = [NSJSONSerialization JSONObjectWithData:data options:0 error:error];
  if (!manifest) return nil;
  try {
    _model = std::make_unique<ModelState>();
    auto &m = *_model;
    m.options.SetIntraOpNumThreads(1);
    m.options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    m.session = std::make_unique<Ort::Session>(m.environment, path.UTF8String, m.options);
    m.window = [manifest[@"window"] longLongValue];
    m.mels = [manifest[@"fbank"][@"mels"] longLongValue];
    m.names.emplace_back("x");
    m.outputs.emplace_back("log_probs");
    for (NSDictionary *state in manifest[@"states"]) {
      std::string name = [state[@"name"] UTF8String];
      m.names.push_back(name);
      m.outputs.push_back("new_" + name);
      std::vector<int64_t> shape;
      for (NSNumber *d in state[@"shape"]) shape.push_back(d.longLongValue);
      m.shapes.push_back(std::move(shape));
      m.integers.push_back([state[@"dtype"] isEqualToString:@"int64"]);
    }
    m.reset();
    return self;
  } catch (const std::exception &e) { assignError(error, e); return nil; }
}

- (NSData *)runFeatures:(NSData *)features error:(NSError **)error {
  try {
    auto &m = *_model;
    if (features.length != m.window * m.mels * sizeof(float)) {
      throw std::runtime_error("Incorrect feature window dimensions");
    }
    int64_t dimensions[] = {1, m.window, m.mels};
    std::vector<Ort::Value> input;
    input.push_back(Ort::Value::CreateTensor<float>(m.memory, (float *)features.bytes,
      features.length / sizeof(float), dimensions, 3));
    for (auto &state : m.states) input.push_back(std::move(state));
    std::vector<const char *> names, outputs;
    for (auto &name : m.names) names.push_back(name.c_str());
    for (auto &name : m.outputs) outputs.push_back(name.c_str());
    auto result = m.session->Run(Ort::RunOptions{nullptr}, names.data(), input.data(),
      input.size(), outputs.data(), outputs.size());
    auto count = result[0].GetTensorTypeAndShapeInfo().GetElementCount();
    NSData *probabilities = [NSData dataWithBytes:result[0].GetTensorData<float>() length:count * sizeof(float)];
    m.states.clear();
    for (size_t i = 1; i < result.size(); ++i) m.states.push_back(std::move(result[i]));
    return probabilities;
  } catch (const std::exception &e) { assignError(error, e); return nil; }
}

- (void)reset { if (_model) _model->reset(); }
@end
