#pragma once
#include "aether/types.hpp"
#include <filesystem>
#include <vector>
namespace aether {
struct LayerWeights {
  std::vector<float> attn_norm, ffn_norm;
  std::vector<uint16_t> wq, wk, wv, wo, w1, w2, w3;
};
struct Model {
  Config config;
  std::vector<uint16_t> embedding, lm_head;
  std::vector<float> final_norm;
  std::vector<LayerWeights> layers;
  static Model load(const std::filesystem::path& path);
  static Model random_for_demo(const Config& config);
  void save(const std::filesystem::path& path) const;
};
}
