#pragma once
#include "aether/types.hpp"
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace aether {
class MetalRuntime {
 public:
  MetalRuntime(); ~MetalRuntime();
  MetalRuntime(const MetalRuntime&) = delete; MetalRuntime& operator=(const MetalRuntime&) = delete;
  bool available() const noexcept;
  std::string device_name() const;
  void gemm_f16(const void* a, const void* b, void* c, uint32_t m, uint32_t n, uint32_t k);
  void gemm_i8(const int8_t* a, const int8_t* b, float* c, uint32_t m, uint32_t n, uint32_t k, float scale_a, float scale_b);
  void rms_norm(const float* x, const float* weight, float* y, uint32_t rows, uint32_t cols, float eps);
  void silu_mul(const float* gate, const float* up, float* out, uint32_t n);
  void attention(const float* q, const float* k, const float* v, float* out,
                 uint32_t seq, uint32_t head_dim, float scale);
  void synchronize();
  double last_kernel_ms() const noexcept;
  std::string profile_report() const;
 private: struct Impl; std::unique_ptr<Impl> p_;
};
}
