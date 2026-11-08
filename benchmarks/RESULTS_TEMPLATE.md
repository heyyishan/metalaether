# Benchmark report template

Do not fill this with estimates. Run on the target Mac and attach the raw CSV and Instruments capture.

| Field | Value |
|---|---|
| Mac / chip | TODO |
| Unified memory | TODO |
| macOS | TODO |
| Xcode / Metal compiler | TODO |
| Build commit / flags | TODO |
| Model / parameters | TODO |
| Weight dtype | TODO |
| Warm-up / measured iterations | TODO |
| Thermal / power state | TODO |

| Prompt length | Batch | Decode tokens | Prefill tok/s | Decode tok/s | Median token latency | p95 token latency | GPU kernel time |
|---:|---:|---:|---:|---:|---:|---:|---:|
| | | | | | | | |

## Kernel breakdown

| Kernel | Calls / token | Median GPU time | Bytes requested | Notes |
|---|---:|---:|---:|---|
| FP16 GEMM | | | | |
| INT8 GEMM | | | | |
| RMSNorm | | | | |
| SiLU × gate | | | | |
| Attention scores | | | | |
| Attention softmax | | | | |
| Attention value | | | | |

## Method

Record warm-up, synchronization, capture settings, and whether GPU timings are from Instruments or host wall-clock. Report physical memory bandwidth only when a hardware counter provides it.
