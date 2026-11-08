# Aether

**An experimental Transformer inference runtime for Apple Silicon, written in C++23 and Metal Shading Language.**

Aether is a deliberately small inference-engine project. Its purpose is to make the performance-critical parts of autoregressive Transformer inference inspectable: matrix multiplication, attention, normalization, activation, weight representation, and KV-cache lifetime. It is not a wrapper around `torch`, Core ML, or llama.cpp.

> **Status: research MVP / kernel prototype.** The repository contains runnable Metal compute kernels and a compact decoder-only inference path. It is not a production LLM runtime, and its current generation path is intentionally conservative. No benchmark numbers are published because this project has not been executed on the target Mac in this environment.

## Why Aether?

Small and medium models are often limited less by theoretical peak FLOP/s than by the cost of moving weights and intermediate activations, launching many small kernels, and synchronizing the CPU with the GPU. Autoregressive decoding is particularly unforgiving: each token has a small amount of arithmetic but repeatedly touches a large fraction of the model's weights.

Aether is organized around three questions:

1. Can the GPU reuse tiles of matrix operands instead of repeatedly fetching them from device memory?
2. Can the runtime keep weights and the KV cache resident in unified memory rather than allocate and copy them for every operation?
3. Can command submission and synchronization overhead be made small compared with useful GPU work?

The first question has a concrete kernel implementation here. The latter two are partially implemented and form the main optimization roadmap.

## Scope

| Capability | MVP status | Notes |
|---|---|---|
| Apple Silicon Metal compute | Implemented | Uses Metal compute pipelines and shared-storage buffers |
| Tiled FP16 GEMM | Implemented | 16 × 16 output tile, threadgroup operand staging |
| INT8 GEMM | Implemented | Integer accumulation, scalar dequantization scales |
| RMSNorm | Implemented | One threadgroup per row, 256-way reduction |
| SiLU × gate | Implemented | Fused elementwise kernel |
| Causal attention | Implemented | Separate score, softmax, and value passes; correctness-oriented baseline |
| KV cache | Basic | Append-only per-layer host-side FP32 vectors |
| Decoder-only generation | Basic | Small custom model format and byte tokenizer |
| Python frontend | Implemented when pybind11 is found | `Model`, `Engine`, `GenerateOptions` |
| Static batching | Not implemented | Batch argument currently rejects values other than one |
| Continuous batching | Not implemented | Scheduler is a future extension |
| FlashAttention-style online softmax | Not implemented | Current attention materializes the score matrix |
| Native Metal 4 command submission | Not implemented | Current backend uses `MTLCommandQueue` / `MTLComputeCommandEncoder` |
| Pretrained Hugging Face model import | Not implemented | The `.aether` format is a small experimental container |

## Platform requirements

- Apple Silicon Mac.
- macOS 15 or later for the current Metal compute backend.
- Xcode Command Line Tools (`xcode-select --install`).
- CMake 3.24+, Apple Clang with C++23 support.
- Python 3.10+ and `pybind11` for the Python extension.
- Optional: PyTorch for the CPU GEMM comparison script.

Metal 4 is available on Apple Silicon with macOS Tahoe 26 or later. The current code uses the established Metal compute API, which remains available on those systems; it does **not** yet use the new `MTL4*` command submission family. See [Apple's Metal 4 core API overview](https://developer.apple.com/documentation/metal/understanding-the-metal-4-core-api) and [Metal device support](https://support.apple.com/en-us/102894).

## Architecture and data flow

```text
Python caller
    │
    ▼
pybind11 module
    │
    ▼
C++ Engine ───── ByteTokenizer
    │
    ├── Model weights (.aether)
    ├── Per-layer append-only K/V cache
    └── Decode loop
          │
          ├── RMSNorm
          ├── Q, K, V projections ── FP16 tiled GEMM
          ├── Causal attention ───── score / softmax / value kernels
          ├── Output projection ──── FP16 tiled GEMM
          ├── RMSNorm + gated MLP ── RMSNorm / GEMM / SiLU×gate / GEMM
          └── Vocabulary projection
                    │
                    ▼
             Metal compute queue
                    │
                    ▼
             Apple GPU / unified memory
```

### Runtime ownership

`Model` owns the model's serialized weight arrays. `Engine` owns the model, the Metal runtime, and the tokenizer. The current KV cache is scoped to one generation request and is append-only for each layer. This makes cache lifetime easy to reason about, but it is not the final high-performance design: the current cache is held in host vectors and uploaded into fresh Metal buffers for attention calls.

### Tensor layout

- Projection weights are row-major `[input_features, output_features]`.
- Activations are row-major `[rows, features]`.
- GEMM computes `C[M,N] = A[M,K] × B[K,N]`.
- The FP16 path stores operands and output as IEEE half; accumulation is FP32.
- The INT8 path uses signed 8-bit operands, int32 accumulation, and two scalar scales.
- Normalization weights and attention intermediates are FP32 in this prototype.

## Kernel design

### 1. Tiled FP16 GEMM

`metal/kernels.metal::gemm_f16_tiled` computes a 16 × 16 output tile per threadgroup. Each iteration stages one 16 × 16 tile of A and B into threadgroup memory, synchronizes, and accumulates the tile product in FP32 registers. Boundary loads are zero-filled, so dimensions need not be multiples of 16.

The main trade-off is deliberate simplicity. A single thread computes one output element, which makes the indexing and memory reuse visible. It is a baseline for tuning, not a claim to match Apple's vendor GEMM implementations. A next version should benchmark several tile shapes, use SIMD-group matrix operations where supported, and specialize decode GEMV separately from prefill GEMM.

### 2. INT8 path

`gemm_i8` performs signed integer dot products and dequantizes the accumulator using `scale_a × scale_b`. It is a minimal symmetric per-tensor quantization path. There are no zero points, per-channel scales, calibration tools, or quantized model importer yet. It exists to establish the interface and arithmetic path, not to claim competitive quantized inference.

### 3. Attention

The attention implementation is intentionally split into three kernels:

1. QKᵀ score calculation with causal masking.
2. Row-wise stable softmax using max subtraction.
3. Weighted value accumulation.

This makes intermediate memory traffic visible and provides a correctness baseline. It is not FlashAttention. The score tensor is materialized in memory and the three dispatches create extra launch and synchronization costs. The next implementation should use online softmax and tiled Q/K/V blocks, avoiding the `sequence_length²` score allocation.

### 4. RMSNorm and gated activation

RMSNorm reduces the squared activations within a threadgroup, computes the inverse root mean square, then applies the learned scale. The gated activation fuses SiLU and elementwise multiplication, avoiding a separate activation output and multiply pass.

## Build

```bash
xcode-select --install
brew install cmake

# Optional Python binding dependency
python3 -m pip install pybind11

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The build compiles `metal/kernels.metal` with `xcrun metal`, links a `default.metallib`, and builds the C++ runtime. If CMake cannot locate pybind11, the C++ executable still builds; install pybind11 and configure again to build the extension.

Run the C++ demo:

```bash
./build/aether
```

With no model path, the executable creates a tiny random-weight model in `demo.aether`. **Random weights are only a runtime smoke test. They do not produce meaningful language.** To load a model:

```bash
./build/aether path/to/model.aether
```

The `.aether` format is currently an experimental binary format written by `Model::save`; it is not compatible with safetensors, GGUF, or Hugging Face checkpoints.

## Python usage

After building the extension, make the package visible to Python. One simple development setup is:

```bash
PYTHONPATH=python:build python3 - <<'PY'
import aether
cfg = aether.Config()
cfg.dim = 128
cfg.hidden_dim = 384
cfg.n_layers = 2
cfg.n_heads = 4
cfg.max_seq_len = 256
model = aether.Model.random_for_demo(cfg)
model.save("demo.aether")
engine = aether.Engine(model)
print("GPU:", engine.device)
print(engine.generate("Aether", aether.GenerateOptions()))
PY
```

Depending on the CMake generator and platform, `_aether` may be emitted in `build/` rather than `python/aether/`. Copy or symlink the built extension into `python/aether/` for the package import layout, or adjust `PYTHONPATH` accordingly.

## Benchmark methodology

The repository deliberately ships **benchmark harnesses, not fabricated benchmark results**. A real result depends on the exact Mac model, GPU core configuration, memory capacity, macOS release, thermal state, compiler, and background load.

### End-to-end generation sweep

Create a demo model (or load an actual `.aether` model), then run:

```bash
PYTHONPATH=python:build python3 scripts/make_demo.py
PYTHONPATH=python:build python3 benchmarks/bench.py \
  --model demo.aether \
  --prompts 32 128 256 \
  --tokens 16 32 64 128 \
  --out aether_bench.csv
```

The current benchmark reports end-to-end generated tokens per second, including prompt processing. It does not yet isolate prefill throughput from decode throughput. The `reported_weight_bytes_per_step` field is the model's nominal FP16 projection-weight footprint, not a hardware counter measurement of bytes transferred from DRAM.

### CPU GEMM baseline

```bash
python3 -m pip install torch
python3 benchmarks/torch_cpu_baseline.py --m 1 --n 4096 --k 4096 --iters 100
python3 benchmarks/torch_cpu_baseline.py --m 128 --n 4096 --k 4096 --iters 100
```

This is a CPU-only PyTorch matrix multiplication microbenchmark. Compare equivalent shapes and dtypes. It is **not** an end-to-end Transformer comparison, and it is not a comparison against MPS. For a fair end-to-end CPU baseline, implement the same model and tokenizer on CPU and report prompt and decode separately.

### Kernel timing and memory bandwidth

The current runtime aggregates elapsed host wait time and dispatch counts by kernel name; the benchmark writes a companion `*_kernel_host_timings.csv`. These measurements include host submission and synchronization overhead; they are not pure GPU execution durations. For kernel-level GPU timings, capture the workload with Xcode's Metal GPU capture / Instruments and record each kernel's GPU duration. For DRAM bandwidth utilization, use hardware performance counters exposed by the relevant Instruments tooling where available. Do not infer physical DRAM traffic from tensor sizes alone: cache hits, compression, residency, and reuse change actual traffic.

A publishable benchmark report should include:

- Exact Mac model / chip and unified memory capacity.
- macOS and Xcode versions, compiler flags, and build commit.
- Model architecture, parameter count, weight dtype, and context length.
- Warm-up count, timed iterations, synchronization policy, and thermal conditions.
- Prompt tokens/s and decode tokens/s separately.
- Batch size and sequence length for every result.
- Median and p95 latency, not just a single mean.
- GPU kernel duration and measured bandwidth where available.

## Engineering trade-offs and known limitations

1. **Buffer allocation and copies:** current dispatch helpers allocate shared Metal buffers per call and copy host arrays into them. This is easy to inspect but defeats much of the benefit of unified memory. Production work needs a persistent tensor allocator, long-lived weight buffers, and cache buffers that grow geometrically.
2. **Synchronization:** each kernel waits synchronously for completion. This is useful for a correctness-first MVP, but incurs CPU/GPU round trips. A command encoder should batch a full layer or decode step before synchronization.
3. **Attention:** materialized score matrices are quadratic in sequence length. Replace with online-softmax tiled attention.
4. **KV cache placement:** the current cache is host-side FP32 and is re-uploaded. Move it to persistent GPU-visible buffers, use a paged or block allocator, and store K/V in FP16 or a supported quantized format.
5. **Decode GEMM:** one-token projections are GEMV-like. A dedicated weight-streaming kernel is likely more appropriate than the general 16 × 16 GEMM kernel.
6. **Batching:** there is no request scheduler or continuous batching yet. The benchmark rejects batch sizes above one rather than pretending to support them.
7. **Model compatibility:** the byte tokenizer and custom model container are development conveniences, not production model formats.
8. **Metal 4 adoption:** port command submission to `MTL4CommandQueue`, reusable `MTL4CommandBuffer` / command allocators, and argument tables. The current implementation is Metal-4-system compatible, not a native Metal 4 API backend.
9. **Numerical validation:** add reference comparisons against PyTorch for every kernel, including odd dimensions, long sequences, and adversarial normalization inputs.

## Roadmap

### P0: correctness and observability
- Add per-kernel numerical tests against PyTorch.
- Separate prefill and decode paths in the benchmark.
- Add median/p95 reporting and GPU capture instructions.
- Validate model file dimensions and reject malformed tensors.

### P1: memory hierarchy
- Persistent weight and activation buffers.
- Persistent GPU-resident KV cache with capacity growth.
- Command-buffer batching to remove per-kernel CPU waits.
- Explicit resource residency and lifetime tracking.

### P2: kernel performance
- Native Metal 4 command queue, command buffer, allocator, and argument-table backend.
- SIMD-group matrix kernels and architecture-specific tile sweeps.
- Decode-specialized GEMV / fused QKV projection.
- Online-softmax tiled attention (FlashAttention-style).
- Fused residual + RMSNorm and fused MLP epilogue.

### P3: model and serving features
- Safetensors importer and tokenizer integration.
- Per-channel INT8 quantization and calibration.
- Static batching, then continuous batching.
- Sampling (temperature, top-k, top-p), stop tokens, and streaming output.

## Repository layout

```text
include/aether/       Public C++ interfaces
src/                  Metal runtime, model format, tokenizer, engine
metal/kernels.metal   GPU compute kernels
python/               pybind11 module and Python package
benchmarks/           End-to-end sweep and CPU GEMM baseline
examples/             C++ generation example
scripts/              Demo model helper
tests/                Model-format smoke test
```

## License

MIT. See [LICENSE](LICENSE).
