#!/usr/bin/env python3
"""Aether throughput sweep. Run on an otherwise idle Mac; results are machine-specific."""
import argparse, csv, platform, time
import aether

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',default='demo.aether'); ap.add_argument('--out',default='aether_bench.csv'); ap.add_argument('--tokens',type=int,nargs='+',default=[16,32,64,128]); ap.add_argument('--prompts',type=int,nargs='+',default=[32,128,256]); args=ap.parse_args()
    model=aether.Model.load(args.model); engine=aether.Engine(model)
    rows=[]
    for p in args.prompts:
        for n in args.tokens:
            r=engine.benchmark(p,n,1); rows.append({'device':engine.device,'macOS':platform.mac_ver()[0],'prompt_tokens':p,'generated_tokens':n,'batch':1,'latency_ms':r.milliseconds,'end_to_end_tok_s':r.tokens_per_second,'reported_weight_bytes_per_step':r.bytes_moved})
            print(rows[-1])
    with open(args.out,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    with open(args.out.replace('.csv','_kernel_host_timings.csv'),'w') as f: f.write(engine.profile_report())
if __name__=='__main__': main()
