#include "aether/model.hpp"
#include <fstream>
#include <random>
#include <stdexcept>
#include <cstring>
#include <bit>
namespace aether {
namespace {
void write_vec(std::ofstream& f,const std::vector<uint16_t>& v){uint64_t n=v.size();f.write((char*)&n,8);f.write((char*)v.data(),n*2);}
void read_vec(std::ifstream& f,std::vector<uint16_t>& v){uint64_t n=0;f.read((char*)&n,8);if(n>1ull<<32)throw std::runtime_error("Invalid tensor size");v.resize(n);f.read((char*)v.data(),n*2);if(!f)throw std::runtime_error("Truncated model file");}
void write_f(std::ofstream& f,const std::vector<float>& v){uint64_t n=v.size();f.write((char*)&n,8);f.write((char*)v.data(),n*4);}
void read_f(std::ifstream& f,std::vector<float>& v){uint64_t n=0;f.read((char*)&n,8);if(n>1ull<<32)throw std::runtime_error("Invalid tensor size");v.resize(n);f.read((char*)v.data(),n*4);if(!f)throw std::runtime_error("Truncated model file");}
}
Model Model::random_for_demo(const Config& c){Model m;m.config=c;std::mt19937 gen(7);std::normal_distribution<float>d(0,0.025f);auto init=[&](std::vector<uint16_t>&v,size_t n){v.resize(n);for(auto&x:v){float z=d(gen);x=std::bit_cast<uint16_t>((_Float16)z);}};
 init(m.embedding,size_t(c.vocab_size)*c.dim);m.final_norm.assign(c.dim,1.f);init(m.lm_head,size_t(c.dim)*c.vocab_size);
 for(uint32_t i=0;i<c.n_layers;++i){LayerWeights w;w.attn_norm.assign(c.dim,1.f);w.ffn_norm.assign(c.dim,1.f);init(w.wq,size_t(c.dim)*c.dim);init(w.wk,size_t(c.dim)*c.dim);init(w.wv,size_t(c.dim)*c.dim);init(w.wo,size_t(c.dim)*c.dim);init(w.w1,size_t(c.dim)*c.hidden_dim);init(w.w2,size_t(c.hidden_dim)*c.dim);init(w.w3,size_t(c.dim)*c.hidden_dim);m.layers.push_back(std::move(w));}return m;}
void Model::save(const std::filesystem::path& p)const{std::ofstream f(p,std::ios::binary);if(!f)throw std::runtime_error("Cannot write model");const char magic[8]={'A','E','T','H','E','R','0','1'};f.write(magic,8);f.write((char*)&config,sizeof(config));write_vec(f,embedding);write_f(f,final_norm);write_vec(f,lm_head);for(auto&w:layers){write_f(f,w.attn_norm);write_f(f,w.ffn_norm);write_vec(f,w.wq);write_vec(f,w.wk);write_vec(f,w.wv);write_vec(f,w.wo);write_vec(f,w.w1);write_vec(f,w.w2);write_vec(f,w.w3);}}
Model Model::load(const std::filesystem::path& p){std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("Cannot open model: "+p.string());char magic[8];f.read(magic,8);if(std::memcmp(magic,"AETHER01",8))throw std::runtime_error("Not an Aether model");Model m;f.read((char*)&m.config,sizeof(m.config));read_vec(f,m.embedding);read_f(f,m.final_norm);read_vec(f,m.lm_head);m.layers.resize(m.config.n_layers);for(auto&w:m.layers){read_f(f,w.attn_norm);read_f(f,w.ffn_norm);read_vec(f,w.wq);read_vec(f,w.wk);read_vec(f,w.wv);read_vec(f,w.wo);read_vec(f,w.w1);read_vec(f,w.w2);read_vec(f,w.w3);}return m;}
}
