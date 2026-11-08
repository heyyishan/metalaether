"""CPU GEMM baseline for the same matrix shapes. Requires PyTorch; deliberately CPU-only."""
import argparse, time, torch

def main():
    p=argparse.ArgumentParser();p.add_argument('--m',type=int,default=1);p.add_argument('--n',type=int,default=4096);p.add_argument('--k',type=int,default=4096);p.add_argument('--iters',type=int,default=100);a=p.parse_args()
    torch.set_num_threads(torch.get_num_threads()); A=torch.randn(a.m,a.k,dtype=torch.float16);B=torch.randn(a.k,a.n,dtype=torch.float16)
    for _ in range(10): torch.matmul(A,B)
    t=time.perf_counter()
    for _ in range(a.iters): torch.matmul(A,B)
    ms=(time.perf_counter()-t)*1000/a.iters
    print({'backend':'PyTorch CPU','threads':torch.get_num_threads(),'shape':(a.m,a.n,a.k),'mean_ms':ms,'gflop_s':(2*a.m*a.n*a.k)/(ms*1e6)})
if __name__=='__main__':main()
