#include "aether/tokenizer.hpp"
namespace aether {
std::vector<uint32_t> ByteTokenizer::encode(const std::string& s)const{std::vector<uint32_t> r;r.reserve(s.size());for(unsigned char c:s)r.push_back(c);return r;}
std::string ByteTokenizer::decode(const std::vector<uint32_t>& ids)const{std::string s;s.reserve(ids.size());for(auto x:ids)if(x<256)s.push_back(static_cast<char>(x));return s;}
}
