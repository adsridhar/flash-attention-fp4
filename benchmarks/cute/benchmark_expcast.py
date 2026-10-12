"""Compare prepared HAO base and production ExpCast kernels on Blackwell."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoding', choices=('both', 'standard', 'expcast'), default='both')
    parser.add_argument('--q-len', type=int, default=44160)
    parser.add_argument('--kv-len', type=int, default=46080)
    parser.add_argument('--q-heads', type=int, default=32)
    parser.add_argument('--kv-heads', type=int, default=8)
    parser.add_argument('--threshold', type=float, default=8.0)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--repetitions', type=int, default=50)
    parser.add_argument('--seed', type=int, default=20260921)
    parser.add_argument('--output', type=Path, default=ROOT / 'agent_space/expcast_benchmark/results.json')
    parser.add_argument('--save-output', action='store_true', help='Save each BF16 output beside the JSON.')
    args = parser.parse_args()
    if args.q_len <= 0 or args.kv_len <= 0 or args.q_len % 128 or args.kv_len % 128:
        parser.error('Q and KV lengths must be positive multiples of 128')
    if args.kv_heads <= 0 or args.q_heads <= 0 or args.q_heads % args.kv_heads:
        parser.error('Q heads must be a positive multiple of positive KV heads')
    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 8:
        parser.error('Threshold must be between 0 and 8 for this unclamped, offset-0 benchmark')
    if args.warmup < 1 or args.repetitions < 1:
        parser.error('Warmup and repetitions must be positive')
    return args


def configure(threshold):
    # Set before importing/compiling either kernel. Disable experimental variants.
    settings = {
        'RESCALE_THRESHOLD': str(threshold), 'P_LOG2_OFFSET': '0',
        'EXPCAST_BETA': '-0.35', 'EXPCAST_MAX_CODE': '126',
        'EXPCAST_NATIVE_DECODE': '1', 'EXPCAST_STREAM_SUM': '1',
        'EXPCAST_PACKED_MAPPING': '1', 'EXPCAST_PACKED_CONVERSION': '1',
        'EXPCAST_REMOVE_CLAMPS': '1', 'EXPCAST_EARLY_PUBLISH': '0',
        'EXPCAST_PROFILE_SPLIT': '0', 'EXPCAST_CORRECTION_DECODE': '0',
        'EXPCAST_CORRECTION_DECODE_SPLIT': '0',
    }
    for name, value in settings.items():
        os.environ['FA4_FP8_PV_' + name] = value


def quantize_k(torch, quantize, layout, rows):
    """Build NVFP4 K codes and the kernel's block-16 scale layout."""
    batch, length, heads, dim = rows.shape
    codes, scales = quantize(
        rows.reshape(batch * length, heads * dim),
        torch.ones(1, device=rows.device, dtype=torch.float32),
        sfLayout=layout, do_shuffle=False,
    )
    codes = codes.reshape(batch, length, heads, dim // 2).view(torch.uint8).view(torch.float4_e2m1fn_x2)
    scales = (scales.reshape(batch, length // 128, heads, dim // 64, 32, 4, 4)
              .permute(0, 2, 1, 3, 4, 5, 6).contiguous().permute(4, 5, 2, 6, 3, 1, 0))
    return codes, scales


def quantize_q(torch, quantize, layout, rows, kv_heads):
    """Quantize in packed-GQA row order; restore Q codes to BSHD."""
    batch, length, heads, dim = rows.shape
    ratio = heads // kv_heads
    source = (rows.reshape(batch, length, kv_heads, ratio, dim)
              .permute(0, 2, 1, 3, 4).contiguous().reshape(-1, dim))
    codes, scales = quantize(
        source, torch.ones(1, device=rows.device, dtype=torch.float32),
        sfLayout=layout, do_shuffle=False,
    )
    codes = (codes.reshape(batch, kv_heads, length, ratio, dim // 2)
             .permute(0, 2, 1, 3, 4).reshape(batch, length, heads, dim // 2)
             .contiguous().view(torch.uint8).view(torch.float4_e2m1fn_x2))
    scales = scales.reshape(batch, kv_heads, length * ratio // 128, dim // 64, 32, 4, 4).permute(4, 5, 2, 6, 3, 1, 0)
    return codes, scales


def measure(torch, launch, warmup, repetitions):
    # Lazy compilation and warmup are outside every timing sample.
    for _ in range(warmup):
        output = launch()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(repetitions):
        start.record()
        output = launch()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    if not torch.isfinite(output).all().item():
        raise RuntimeError('Attention returned nonfinite values')
    return output, {'median_ms': statistics.median(samples), 'min_ms': min(samples),
                    'max_ms': max(samples), 'samples_ms': samples}


def main():
    args = parse_args()
    configure(args.threshold)
    import torch
    import cutlass.cute as cute
    from flashinfer.quantization import SfLayout, nvfp4_quantize
    # Compatibility with CUTLASS DSL versions that renamed make_fragment.
    if not hasattr(cute, 'make_fragment'):
        cute.make_fragment = cute.make_rmem_tensor
    from flash_attn.cute import interface

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError('This benchmark requires a Blackwell SM100-family GPU')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    q = torch.randn((1, args.q_len, args.q_heads, 128), device='cuda', dtype=torch.bfloat16)
    k = torch.randn((1, args.kv_len, args.kv_heads, 128), device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k).to(torch.float8_e4m3fn)
    q, sfq = quantize_q(torch, nvfp4_quantize, SfLayout.layout_128x4, q, args.kv_heads)
    k, sfk = quantize_k(torch, nvfp4_quantize, SfLayout.layout_128x4, k)

    def launch():
        output = interface.flash_attn_func(
            q, k, v, causal=False, softmax_scale=128 ** -0.5,
            pack_gqa=True, mSFQ=sfq, mSFK=sfk,
        )
        return output[0] if isinstance(output, tuple) else output

    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    result = {
        'hardware': {'gpu': device.name, 'sm_count': device.multi_processor_count,
                     'compute_capability': list(torch.cuda.get_device_capability()),
                     'torch': torch.__version__, 'torch_cuda': torch.version.cuda},
        'shape': {'batch': 1, 'q_len': args.q_len, 'kv_len': args.kv_len,
                  'q_heads': args.q_heads, 'kv_heads': args.kv_heads, 'head_dim': 128},
        'protocol': {'warmup': args.warmup, 'repetitions': args.repetitions, 'seed': args.seed,
                     'scope': 'CUDA-event time per prepared attention invocation; quantization and compilation excluded'},
        'config': {'qk': 'NVFP4 block-16 scales', 'pv': 'E4M3 FP8', 'pack_gqa': True,
                   'threshold': args.threshold, 'p_log2_offset': 0, 'beta': -0.35, 'max_code': 126,
                   'native_decode': True, 'stream_sum': True, 'early_publish': False},
        'kernels': {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encodings = ('standard', 'expcast') if args.encoding == 'both' else (args.encoding,)
    for encoding in encodings:
        os.environ['FA4_FP8_PV_P_ENCODING'] = encoding
        output, timing = measure(torch, launch, args.warmup, args.repetitions)
        result['kernels'][encoding] = timing
        if args.save_output:
            path = args.output.with_name(args.output.stem + '_' + encoding + '.pt')
            torch.save(output.cpu(), path)
            timing['output_tensor'] = str(path)
        print(f'{encoding:8s}: {timing["median_ms"]:.3f} ms', flush=True)
    if args.encoding == 'both':
        speedup = result['kernels']['standard']['median_ms'] / result['kernels']['expcast']['median_ms']
        result['expcast_speedup_vs_standard'] = speedup
        print(f'ExpCast kernel speedup: {speedup:.3f}x', flush=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Results: {args.output}')


if __name__ == '__main__':
    main()
