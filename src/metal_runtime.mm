#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "aether/metal_runtime.hpp"
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <sstream>
#include <iomanip>

namespace aether {
struct MetalRuntime::Impl {
  id<MTLDevice> device=nil; id<MTLCommandQueue> queue=nil; id<MTLLibrary> library=nil;
  std::unordered_map<std::string,id<MTLComputePipelineState>> pipelines;
  std::unordered_map<std::string,std::pair<double,uint64_t>> timings;
  double last_ms=0;
  id<MTLComputePipelineState> pipeline(const char* name) {
    auto it=pipelines.find(name); if(it!=pipelines.end())return it->second;
    NSError* error=nil; id<MTLFunction> fn=[library newFunctionWithName:@(name)];
    if(!fn) throw std::runtime_error(std::string("Missing Metal function: ")+name);
    id<MTLComputePipelineState> p=[device newComputePipelineStateWithFunction:fn error:&error];
    if(!p) throw std::runtime_error([[error localizedDescription] UTF8String]); pipelines.emplace(name,p); return p;
  }
  id<MTLBuffer> buffer(const void* data,size_t bytes) {
    id<MTLBuffer> b=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if(!b)throw std::runtime_error("Metal buffer allocation failed"); if(data&&bytes)memcpy([b contents],data,bytes); return b;
  }
  template<class F> void dispatch(const char* name, MTLSize grid, MTLSize group, F bind) {
    id<MTLCommandBuffer> cb=[queue commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    [e setComputePipelineState:pipeline(name)]; bind(e); [e dispatchThreads:grid threadsPerThreadgroup:group]; [e endEncoding];
    auto t=std::chrono::steady_clock::now(); [cb commit]; [cb waitUntilCompleted];
    last_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();
    auto& stat=timings[name]; stat.first+=last_ms; stat.second++;
    if(cb.status==MTLCommandBufferStatusError) throw std::runtime_error([[cb.error localizedDescription] UTF8String]);
  }
};
MetalRuntime::MetalRuntime():p_(std::make_unique<Impl>()) {
  p_->device=MTLCreateSystemDefaultDevice(); if(!p_->device)throw std::runtime_error("No Metal device found");
  p_->queue=[p_->device newCommandQueue]; if(!p_->queue)throw std::runtime_error("Unable to create Metal command queue");
  NSError* error=nil; NSString* path=@(AETHER_METALLIB_PATH); p_->library=[p_->device newLibraryWithFile:path error:&error];
  if(!p_->library)throw std::runtime_error([[error localizedDescription] UTF8String]);
}
MetalRuntime::~MetalRuntime()=default;
bool MetalRuntime::available()const noexcept{return p_&&p_->device;}
std::string MetalRuntime::device_name()const{return p_?[[p_->device name] UTF8String]:"unavailable";}
double MetalRuntime::last_kernel_ms()const noexcept{return p_->last_ms;}
std::string MetalRuntime::profile_report()const{std::ostringstream out;out<<"kernel,dispatches,total_host_wait_ms,mean_host_wait_ms\n";for(const auto& [name,v]:p_->timings)out<<name<<","<<v.second<<","<<std::fixed<<std::setprecision(4)<<v.first<<","<<(v.second?v.first/v.second:0)<<"\n";return out.str();}
void MetalRuntime::synchronize(){id<MTLCommandBuffer> cb=[p_->queue commandBuffer];[cb commit];[cb waitUntilCompleted];}
void MetalRuntime::gemm_f16(const void* a,const void* b,void* c,uint32_t m,uint32_t n,uint32_t k){
  auto A=p_->buffer(a,size_t(m)*k*2),B=p_->buffer(b,size_t(k)*n*2),C=p_->buffer(nullptr,size_t(m)*n*2);
  p_->dispatch("gemm_f16_tiled",MTLSizeMake(n,m,1),MTLSizeMake(16,16,1),[&](id<MTLComputeCommandEncoder> e){
    uint32_t M=m,N=n,K=k;[e setBuffer:A offset:0 atIndex:0];[e setBuffer:B offset:0 atIndex:1];[e setBuffer:C offset:0 atIndex:2];[e setBytes:&M length:4 atIndex:3];[e setBytes:&N length:4 atIndex:4];[e setBytes:&K length:4 atIndex:5];});
  memcpy(c,[C contents],size_t(m)*n*2);
}
void MetalRuntime::gemm_i8(const int8_t* a,const int8_t* b,float* c,uint32_t m,uint32_t n,uint32_t k,float sa,float sb){
  auto A=p_->buffer(a,size_t(m)*k),B=p_->buffer(b,size_t(k)*n),C=p_->buffer(nullptr,size_t(m)*n*4);
  p_->dispatch("gemm_i8",MTLSizeMake(n,m,1),MTLSizeMake(8,8,1),[&](id<MTLComputeCommandEncoder> e){uint32_t M=m,N=n,K=k;[e setBuffer:A offset:0 atIndex:0];[e setBuffer:B offset:0 atIndex:1];[e setBuffer:C offset:0 atIndex:2];[e setBytes:&M length:4 atIndex:3];[e setBytes:&N length:4 atIndex:4];[e setBytes:&K length:4 atIndex:5];[e setBytes:&sa length:4 atIndex:6];[e setBytes:&sb length:4 atIndex:7];});memcpy(c,[C contents],size_t(m)*n*4);
}
void MetalRuntime::rms_norm(const float* x,const float* w,float* y,uint32_t rows,uint32_t cols,float eps){auto X=p_->buffer(x,size_t(rows)*cols*4),W=p_->buffer(w,size_t(cols)*4),Y=p_->buffer(nullptr,size_t(rows)*cols*4);p_->dispatch("rms_norm_f32",MTLSizeMake(256,rows,1),MTLSizeMake(256,1,1),[&](id<MTLComputeCommandEncoder>e){[e setBuffer:X offset:0 atIndex:0];[e setBuffer:W offset:0 atIndex:1];[e setBuffer:Y offset:0 atIndex:2];[e setBytes:&cols length:4 atIndex:3];[e setBytes:&eps length:4 atIndex:4];});memcpy(y,[Y contents],size_t(rows)*cols*4);}
void MetalRuntime::silu_mul(const float*g,const float*u,float*o,uint32_t n){auto G=p_->buffer(g,size_t(n)*4),U=p_->buffer(u,size_t(n)*4),O=p_->buffer(nullptr,size_t(n)*4);p_->dispatch("silu_mul_f32",MTLSizeMake(n,1,1),MTLSizeMake(256,1,1),[&](id<MTLComputeCommandEncoder>e){[e setBuffer:G offset:0 atIndex:0];[e setBuffer:U offset:0 atIndex:1];[e setBuffer:O offset:0 atIndex:2];[e setBytes:&n length:4 atIndex:3];});memcpy(o,[O contents],size_t(n)*4);}
void MetalRuntime::attention(const float*q,const float*k,const float*v,float*out,uint32_t seq,uint32_t d,float scale){auto Q=p_->buffer(q,size_t(seq)*d*4),K=p_->buffer(k,size_t(seq)*d*4),V=p_->buffer(v,size_t(seq)*d*4),S=p_->buffer(nullptr,size_t(seq)*seq*4),O=p_->buffer(nullptr,size_t(seq)*d*4);p_->dispatch("attention_scores_f32",MTLSizeMake(seq,seq,1),MTLSizeMake(8,8,1),[&](id<MTLComputeCommandEncoder>e){[e setBuffer:Q offset:0 atIndex:0];[e setBuffer:K offset:0 atIndex:1];[e setBuffer:S offset:0 atIndex:2];[e setBytes:&seq length:4 atIndex:3];[e setBytes:&d length:4 atIndex:4];[e setBytes:&scale length:4 atIndex:5];});p_->dispatch("attention_softmax_f32",MTLSizeMake(seq,1,1),MTLSizeMake(1,1,1),[&](id<MTLComputeCommandEncoder>e){[e setBuffer:S offset:0 atIndex:0];[e setBytes:&seq length:4 atIndex:1];});p_->dispatch("attention_value_f32",MTLSizeMake(d,seq,1),MTLSizeMake(8,8,1),[&](id<MTLComputeCommandEncoder>e){[e setBuffer:S offset:0 atIndex:0];[e setBuffer:V offset:0 atIndex:1];[e setBuffer:O offset:0 atIndex:2];[e setBytes:&seq length:4 atIndex:3];[e setBytes:&d length:4 atIndex:4];});memcpy(out,[O contents],size_t(seq)*d*4);}
}
