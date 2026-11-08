#include "aether/engine.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <chrono>
#include <numeric>
#include <stdexcept>
namespace aether {
namespace {
uint16_t h(float x){return std::bit_cast<uint16_t>((_Float16)x);} float f(uint16_t x){return (float)std::bit_cast<_Float16>(x);}
std::vector<float> dequant(const std::vector<uint16_t>& x){std::vector<float> y(x.size());std::transform(x.begin(),x.end(),y.begin(),f);return y;}
std::vector<uint16_t> quant(const std::vector<float>& x){std::vector<uint16_t> y(x.size());std::transform(x.begin(),x.end(),y.begin(),h);return y;}
std::vector<float> matmul(MetalRuntime& gpu,const std::vector<float>& x,const std::vector<uint16_t>& w,uint32_t m,uint32_t n,uint32_t k){auto a=quant(x);std::vector<uint16_t> c(size_t(m)*n);gpu.gemm_f16(a.data(),w.data(),c.data(),m,n,k);return dequant(c);}
void add(std::vector<float>&a,const std::vector<float>&b){for(size_t i=0;i<a.size();++i)a[i]+=b[i];}
}
Engine::Engine(Model m):model_(std::move(m)),gpu_(){if(model_.config.dim%model_.config.n_heads)throw std::runtime_error("dim must be divisible by n_heads");}
std::string Engine::device()const{return gpu_.device_name();}
std::string Engine::profile_report()const{return gpu_.profile_report();}
std::string Engine::generate(const std::string& prompt,const GenerateOptions& opt){
 const auto& c=model_.config;auto ids=tokenizer_.encode(prompt);if(ids.empty())ids.push_back(1);if(ids.size()>c.max_seq_len)ids.erase(ids.begin(),ids.end()-c.max_seq_len);
 // KV cache is append-only per layer. The cache stores the post-projection K/V vectors in FP32.
 std::vector<std::vector<float>> kc(c.n_layers),vc(c.n_layers);std::vector<uint32_t> out=ids;
 auto step=[&](uint32_t token)->uint32_t{std::vector<float>x(c.dim);for(uint32_t j=0;j<c.dim;++j)x[j]=f(model_.embedding[size_t(token%c.vocab_size)*c.dim+j]);
  for(uint32_t li=0;li<c.n_layers;++li){const auto&w=model_.layers[li];std::vector<float> norm(c.dim);gpu_.rms_norm(x.data(),w.attn_norm.data(),norm.data(),1,c.dim,c.norm_eps);
   auto q=matmul(gpu_,norm,w.wq,1,c.dim,c.dim),k=matmul(gpu_,norm,w.wk,1,c.dim,c.dim),v=matmul(gpu_,norm,w.wv,1,c.dim,c.dim);kc[li].insert(kc[li].end(),k.begin(),k.end());vc[li].insert(vc[li].end(),v.begin(),v.end());
   uint32_t seq=static_cast<uint32_t>(kc[li].size()/c.dim), hd=c.dim/c.n_heads;std::vector<float> last(c.dim,0.0f);
   // Gather each head's strided K/V cache into contiguous head-major scratch, run causal attention,
   // and scatter the final query row back into the concatenated head representation.
   for(uint32_t head=0;head<c.n_heads;++head){std::vector<float> qfull(size_t(seq)*hd,0),kh(size_t(seq)*hd),vh(size_t(seq)*hd),oh(size_t(seq)*hd);std::copy(q.begin()+head*hd,q.begin()+(head+1)*hd,qfull.end()-hd);
    for(uint32_t t=0;t<seq;++t)for(uint32_t d=0;d<hd;++d){kh[size_t(t)*hd+d]=kc[li][size_t(t)*c.dim+head*hd+d];vh[size_t(t)*hd+d]=vc[li][size_t(t)*c.dim+head*hd+d];}
    gpu_.attention(qfull.data(),kh.data(),vh.data(),oh.data(),seq,hd,1.0f/std::sqrt((float)hd));std::copy(oh.end()-hd,oh.end(),last.begin()+head*hd);
   }
   auto proj=matmul(gpu_,last,w.wo,1,c.dim,c.dim);add(x,proj);
   gpu_.rms_norm(x.data(),w.ffn_norm.data(),norm.data(),1,c.dim,c.norm_eps);auto gate=matmul(gpu_,norm,w.w1,1,c.hidden_dim,c.dim),up=matmul(gpu_,norm,w.w3,1,c.hidden_dim,c.dim);std::vector<float> ff(c.hidden_dim);gpu_.silu_mul(gate.data(),up.data(),ff.data(),c.hidden_dim);auto down=matmul(gpu_,ff,w.w2,1,c.dim,c.hidden_dim);add(x,down);
  }
  std::vector<float> norm(c.dim);gpu_.rms_norm(x.data(),model_.final_norm.data(),norm.data(),1,c.dim,c.norm_eps);
  auto logits=matmul(gpu_,norm,model_.lm_head,1,c.vocab_size,c.dim);return static_cast<uint32_t>(std::max_element(logits.begin(),logits.end())-logits.begin());
 };
 for(size_t i=0;i+1<ids.size();++i)(void)step(ids[i]);uint32_t next=step(ids.back());for(uint32_t i=0;i<opt.max_new_tokens&&out.size()<c.max_seq_len;++i){out.push_back(next);if(i+1<opt.max_new_tokens&&out.size()<c.max_seq_len)next=step(next);}return tokenizer_.decode(out);
}
BenchResult Engine::benchmark(uint32_t prompt,uint32_t generated,uint32_t batch){if(batch!=1)throw std::runtime_error("MVP benchmark currently supports batch=1; scheduler interface is reserved for continuous batching");std::string p(prompt,'a');auto t0=std::chrono::steady_clock::now();auto s=generate(p,{generated,0,1});(void)s;auto t1=std::chrono::steady_clock::now();double ms=std::chrono::duration<double,std::milli>(t1-t0).count();uint64_t bytes=0;for(auto&w:model_.layers)bytes+=2ull*(w.wq.size()+w.wk.size()+w.wv.size()+w.wo.size()+w.w1.size()+w.w2.size()+w.w3.size());return{ms,generated/(ms/1000.0),bytes};}
}
