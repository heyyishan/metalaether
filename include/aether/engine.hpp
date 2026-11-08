#pragma once
#include "aether/metal_runtime.hpp"
#include "aether/model.hpp"
#include "aether/tokenizer.hpp"
#include <memory>
#include <string>
namespace aether {
class Engine {
 public:
  explicit Engine(Model model);
  std::string generate(const std::string& prompt, const GenerateOptions& options = {});
  BenchResult benchmark(uint32_t prompt_tokens, uint32_t new_tokens, uint32_t batch = 1);
  std::string device() const;
  std::string profile_report() const;
 private:
  Model model_; MetalRuntime gpu_; ByteTokenizer tokenizer_;
};
}
