# ExpCast kernel benchmark

`benchmarks/cute/benchmark_expcast.py` compares the original HAO kernel with the production ExpCast kernel. It runs directly from this fork and does not import `vllm-omni`.

## Environment

Use a Blackwell B200/B300 GPU and an environment containing PyTorch with `torch.float4_e2m1fn_x2`, CUTLASS DSL, FlashInfer, and this fork's other dependencies. The existing `vllm-omni/.venv` provides these dependencies. The runner puts this checkout first on Python's import path, so the kernel and helpers come from this fork rather than an installed `flash_attn` package.

For the current node:

```bash
cd /home/scratch.adsridhar_other/vllm-omni/fp4-fa4/third_party/hao_flash_attention_fp4_cutlass45
PYTHON=/home/scratch.adsridhar_other/vllm-omni/.venv/bin/python
export CUDA_HOME=/home/scratch.svc_compute_arch/release/cuda_toolkit/internal/latest
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0
```

## Compare the full workload

```bash
"$PYTHON" benchmarks/cute/benchmark_expcast.py \
  --encoding both --q-len 44160 --kv-len 46080 \
  --warmup 20 --repetitions 50 \
  --output agent_space/expcast_benchmark/full.json --save-output
```

The runner prints the two median kernel times and `standard_ms / expcast_ms`. Quantization and compilation are outside timing. Each CUDA-event sample measures one prepared attention invocation, including the interface's output allocation/launch work; this is not BF16-input E2E timing. JSON records all samples, hardware, shape, settings and the speedup. `--save-output` saves the BF16 output tensors as `full_standard.pt` and `full_expcast.pt` beside the JSON.

The two kernels reuse exactly the same quantized inputs. Comparing them in one process ensures matched inputs and hardware.

## Run either kernel separately

```bash
# HAO base: flash_fwd_sm100_fp4.py
"$PYTHON" benchmarks/cute/benchmark_expcast.py --encoding standard \
  --output agent_space/expcast_benchmark/base.json

# Production ExpCast: flash_fwd_sm100_fp4_expcast.py
"$PYTHON" benchmarks/cute/benchmark_expcast.py --encoding expcast \
  --output agent_space/expcast_benchmark/expcast.json
```

For a short smoke run:

```bash
"$PYTHON" benchmarks/cute/benchmark_expcast.py \
  --encoding both --q-len 1024 --kv-len 1024 \
  --warmup 3 --repetitions 10 \
  --output agent_space/expcast_benchmark/small.json
```

Use the full workload to assess production speedup; Q1024/KV1024 has substantially different startup/drain and pipeline behavior.

## Matched settings

Defaults are B1, Q44160/KV46080, Hq32/Hkv8, D128, packed GQA, noncausal attention, no Q smoothing, NVFP4 Q/K with block-16 E4M3 scales, and ordinary E4M3 FP8 V/PV.

Both kernels use **threshold 8 and P log2 offset 0**, explicitly set by the runner. The fork's normal D128 P-offset default is 8, so this benchmark intentionally overrides it for the matched comparison.

- Base generates P with exponentiation and sums its denominator before FP8 conversion.
- ExpCast replaces all per-score P-generation exponentials with affine mapping, integer rounding and packed FP8 encoding. It natively decodes FP8 via FP16 to FP32 and uses streaming denominator accumulation. Beta is −0.35; code cap is 126.
- Per-row online-softmax rescaling still uses `exp2`. Threshold 8 keeps the previous maximum and alpha=1 until the candidate maximum rises by more than 8 log2 units. `--threshold 0` requests conventional maximum updates. The runner restricts the threshold to [0,8] for its offset-0 encoding configuration.
- Experimental early publication, correction-owned decoding, split correction/decode, and split-phase instrumentation are disabled explicitly.

This runner measures performance and checks finite outputs. It does not compute accuracy against cuDNN. Base and ExpCast outputs need not match, because probability approximation and denominator quantization differ. Save outputs for a separate accuracy comparison.
