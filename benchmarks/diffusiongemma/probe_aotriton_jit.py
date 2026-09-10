"""Direct AOTriton 512-head experiment, with an independent FP32 attention oracle.

Use kernels exported from AOTriton 6e00ef3e335b45dfb49065259533b59c68995bfe.
In that export, enable IS_JIT_COMPILING in fwd_kernel_inner.py; this selects
the upstream constexpr argument annotations for direct JIT use. Do not alter
the installed library. See README.md for the experiment's limits.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bm', type=int, default=16)
    parser.add_argument('--bn', type=int, default=16)
    parser.add_argument('--warps', type=int, default=4)
    parser.add_argument('--kv', type=int, default=1280)
    parser.add_argument('--q', type=int, default=256)
    parser.add_argument('--dtype', choices=['bf16', 'fp16'], default='bf16')
    args = parser.parse_args()
    if args.q <= 0 or args.kv <= 0:
        parser.error('sequence lengths must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.kernel_dir.resolve()))
    from fwd_kernel_inner import IS_JIT_COMPILING
    if not IS_JIT_COMPILING:
        raise RuntimeError('enable IS_JIT_COMPILING in the exported kernel source')
    from fwd_kernel import attn_fwd

    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError('requires ROCm PyTorch and a visible GPU')
    torch.manual_seed(42)
    dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float16
    source_hash = hashlib.sha256()
    for path in sorted(args.kernel_dir.glob('*.py')):
        source_hash.update(path.name.encode())
        source_hash.update(path.read_bytes())
    metadata = dict(torch=torch.__version__, hip=torch.version.hip,
                    gpu=torch.cuda.get_device_name(), kernel_sha256=source_hash.hexdigest(),
                    bm=args.bm, bn=args.bn, warps=args.warps, q_len=args.q, kv_len=args.kv,
                    dtype=args.dtype, heads=16, kv_heads=2, head_dim=512)
    print(json.dumps(metadata), flush=True)
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    with torch.inference_mode():
        for magnitude in [0.1, 1.0]:
            q = torch.randn(1, args.q, 16, 512, device='cuda', dtype=dtype).transpose(1, 2) * magnitude
            k = torch.randn(1, 2, args.kv, 512, device='cuda', dtype=dtype) * magnitude
            v = torch.randn_like(k)
            out = torch.full(q.shape, float('nan'), device='cuda', dtype=dtype)
            lse = torch.empty((16, args.q), device='cuda', dtype=torch.float32)
            kwargs = dict(Q=q, K=k, V=v, B=None, A=None, Sm_scale=1.0, L=lse, Out=out,
                          Q_descale=False, K_descale=False, P_scale=False, P_descale=False, V_descale=False,
                          stride_bz=0, stride_bh=0, stride_bm=0, stride_bn=0, stride_az=0, stride_ah=0,
                          Num_head_q=16, Num_head_k=2, Num_seqlens=0, cu_seqlens_q=None, cu_seqlens_k=None,
                          Max_seqlen_q=args.q, Max_seqlen_k=args.kv, seq_strides_q=None, seq_strides_k=None,
                          BLOCK_DMODEL=512, Hdim_qk=512, Hdim_vo=512, PADDED_HEAD=False,
                          ENABLE_DROPOUT=False, dropout_p=0.0, philox_seed_ptr=None, philox_offset1=None,
                          philox_offset2=0, philox_seed_output=None, philox_offset_output=None,
                          RETURN_ENCODED_SOFTMAX=False, encoded_softmax=None, CAUSAL_TYPE=0,
                          Window_left=-1, Window_right=-1, BIAS_TYPE=0, USE_ALIBI=False,
                          INT8=False, INT8_KV=False, USE_P_SCALE=False, PERSISTENT_TYPE=0,
                          persistent_atomic_counter=None, Num_CU=40, GRID_CU_MULTIP=1, Batch=1,
                          BLOCK_M=args.bm, BLOCK_N=args.bn, PRE_LOAD_V=False, NUM_XCDS=1)
            for tensor, names in [(q, ['stride_qz', 'stride_qh', 'stride_qm', 'stride_qk']),
                                  (k, ['stride_kz', 'stride_kh', 'stride_kn', 'stride_kk']),
                                  (v, ['stride_vz', 'stride_vh', 'stride_vk', 'stride_vn']),
                                  (out, ['stride_oz', 'stride_oh', 'stride_om', 'stride_on'])]:
                kwargs.update(zip(names, tensor.stride()))
            run = lambda: attn_fwd[((args.q + args.bm - 1) // args.bm, 16, 1)](
                **kwargs, num_warps=args.warps, num_stages=1, waves_per_eu=1)
            kernel = run()
            torch.cuda.synchronize()
            with sdpa_kernel(SDPBackend.MATH):
                ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(),
                                                    scale=1.0, enable_gqa=True)
                baseline = F.scaled_dot_product_attention(q, k, v, scale=1.0, enable_gqa=True)
            error = (out.float() - ref).abs().max().item()
            row = dict(magnitude=magnitude, max_abs_error=error,
                       finite=bool(out.isfinite().all()),
                       registers=getattr(kernel, 'n_regs', None), spills=getattr(kernel, 'n_spills', None),
                       shared_bytes=kernel.metadata.shared,
                       math_max_abs_error=(baseline.float() - ref).abs().max().item(),
                       rms_error=(out.float() - ref).square().mean().sqrt().item(),
                       math_rms_error=(baseline.float() - ref).square().mean().sqrt().item())
            for name in ['amdgcn', 'llir']:
                if name in kernel.asm:
                    (args.output / f'kernel.{name}').write_text(kernel.asm[name])
            print(json.dumps(row), flush=True)
            (args.output / f'accuracy-{magnitude}.json').write_text(json.dumps(row, indent=2))
            torch.testing.assert_close(out.float(), ref, atol=0.003, rtol=0.02)
            for _ in range(3):
                run()
            torch.cuda.synchronize()
            times = []
            math_times = []
            with sdpa_kernel(SDPBackend.MATH):
                for _ in range(3):
                    F.scaled_dot_product_attention(q, k, v, scale=1.0, enable_gqa=True)
            for _ in range(10):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                run()
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end))
                start.record()
                with sdpa_kernel(SDPBackend.MATH):
                    F.scaled_dot_product_attention(q, k, v, scale=1.0, enable_gqa=True)
                end.record()
                end.synchronize()
                math_times.append(start.elapsed_time(end))
            row.update(status='pass', median_ms=statistics.median(times),
                       math_median_ms=statistics.median(math_times))
            (args.output / f'result-{magnitude}.json').write_text(json.dumps(row, indent=2))
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
