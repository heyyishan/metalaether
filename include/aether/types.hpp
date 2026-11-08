#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace aether {
enum class DType : uint32_t { F16 = 0, I8 = 1 };
struct Config {
  uint32_t vocab_size{256}, dim{256}, hidden_dim{768}, n_layers{4}, n_heads{4}, max_seq_len{1024};
  float norm_eps{1.0e-5f};
  DType dtype{DType::F16};
};
struct GenerateOptions { uint32_t max_new_tokens{64}; float temperature{0.0f}; uint32_t top_k{1}; };
struct BenchResult { double milliseconds{0}; double tokens_per_second{0}; uint64_t bytes_moved{0}; };
}
