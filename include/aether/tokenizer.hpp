#pragma once
#include <cstdint>
#include <string>
#include <vector>
namespace aether {
class ByteTokenizer {
 public:
  std::vector<uint32_t> encode(const std::string& text) const;
  std::string decode(const std::vector<uint32_t>& ids) const;
};
}
