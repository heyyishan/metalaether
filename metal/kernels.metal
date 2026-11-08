#include <metal_stdlib>
using namespace metal;

// Row-major C = A[M,K] * B[K,N]. A/B/C are half to reduce unified-memory traffic.
// A 16x16 output tile reuses A and B values through threadgroup memory.
kernel void gemm_f16_tiled(device const half* A [[buffer(0)]],
                           device const half* B [[buffer(1)]],
                           device half* C [[buffer(2)]],
                           constant uint& M [[buffer(3)]], constant uint& N [[buffer(4)]],
                           constant uint& K [[buffer(5)]],
                           uint2 gid [[threadgroup_position_in_grid]],
                           uint2 tid [[thread_position_in_threadgroup]]) {
  constexpr uint T = 16;
  threadgroup half As[T][T]; threadgroup half Bs[T][T];
  const uint row = gid.y*T + tid.y, col = gid.x*T + tid.x;
  float acc = 0.0f;
  for (uint base=0; base<K; base+=T) {
    As[tid.y][tid.x] = (row<M && base+tid.x<K) ? A[row*K+base+tid.x] : half(0);
    Bs[tid.y][tid.x] = (base+tid.y<K && col<N) ? B[(base+tid.y)*N+col] : half(0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk=0; kk<T; ++kk) acc += float(As[tid.y][kk])*float(Bs[kk][tid.x]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (row<M && col<N) C[row*N+col] = half(acc);
}

// Quantized path: int8 dot product accumulated in int32, then dequantized.
kernel void gemm_i8(device const char* A [[buffer(0)]], device const char* B [[buffer(1)]],
                    device float* C [[buffer(2)]], constant uint& M [[buffer(3)]],
                    constant uint& N [[buffer(4)]], constant uint& K [[buffer(5)]],
                    constant float& SA [[buffer(6)]], constant float& SB [[buffer(7)]],
                    uint2 gid [[thread_position_in_grid]]) {
  uint row=gid.y, col=gid.x; if(row>=M || col>=N) return;
  int acc=0; for(uint k=0;k<K;++k) acc += int(A[row*K+k])*int(B[k*N+col]);
  C[row*N+col]=float(acc)*(SA*SB);
}

kernel void rms_norm_f32(device const float* X [[buffer(0)]], device const float* W [[buffer(1)]],
                         device float* Y [[buffer(2)]], constant uint& C [[buffer(3)]],
                         constant float& EPS [[buffer(4)]], uint3 group_pos [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]]) {
  uint row=group_pos.y; threadgroup float partial[256]; float sum=0.0f;
  for(uint i=tid;i<C;i+=256){float x=X[row*C+i];sum+=x*x;}
  partial[tid]=sum; threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint s=128;s>0;s>>=1){if(tid<s)partial[tid]+=partial[tid+s];threadgroup_barrier(mem_flags::mem_threadgroup);}
  float inv=rsqrt(partial[0]/float(C)+EPS);
  for(uint i=tid;i<C;i+=256)Y[row*C+i]=X[row*C+i]*inv*W[i];
}

kernel void silu_mul_f32(device const float* G [[buffer(0)]], device const float* U [[buffer(1)]],
                         device float* O [[buffer(2)]], constant uint& N [[buffer(3)]],
                         uint id [[thread_position_in_grid]]) {
  if(id<N){float x=G[id];O[id]=(x/(1.0f+exp(-x)))*U[id];}
}

// Single-head causal attention reference kernel. One thread computes one query row/key pair
// in two passes; intended as a correctness baseline before online-softmax tiled attention.
kernel void attention_scores_f32(device const float* Q [[buffer(0)]], device const float* K [[buffer(1)]],
                                 device float* S [[buffer(2)]], constant uint& SEQ [[buffer(3)]],
                                 constant uint& D [[buffer(4)]], constant float& SCALE [[buffer(5)]],
                                 uint2 p [[thread_position_in_grid]]) {
  uint q=p.y, k=p.x; if(q>=SEQ || k>q) return;
  float v=0; for(uint d=0;d<D;++d)v+=Q[q*D+d]*K[k*D+d]; S[q*SEQ+k]=v*SCALE;
}
kernel void attention_softmax_f32(device float* S [[buffer(0)]], constant uint& SEQ [[buffer(1)]],
                                  uint q [[thread_position_in_grid]]) {
  if(q>=SEQ)return; float mx=-INFINITY;
  for(uint k=0;k<=q;++k)mx=max(mx,S[q*SEQ+k]); float z=0;
  for(uint k=0;k<=q;++k){float e=exp(S[q*SEQ+k]-mx);S[q*SEQ+k]=e;z+=e;}
  for(uint k=0;k<=q;++k)S[q*SEQ+k]/=z;
}
kernel void attention_value_f32(device const float* S [[buffer(0)]], device const float* V [[buffer(1)]],
                                device float* O [[buffer(2)]], constant uint& SEQ [[buffer(3)]],
                                constant uint& D [[buffer(4)]], uint2 p [[thread_position_in_grid]]) {
  uint q=p.y,d=p.x;if(q>=SEQ||d>=D)return;float x=0;
  for(uint k=0;k<=q;++k)x+=S[q*SEQ+k]*V[k*D+d];O[q*D+d]=x;
}
