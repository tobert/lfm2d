"""Probe forced ROCm Flash SDPA at DiffusionGemma canvas shapes; no math fallback."""

import json
import statistics

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def project(q, k, v, backend):
    with sdpa_kernel(backend):
        return F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False, scale=1.0, enable_gqa=True)


def time_ms(q, k, v, backend):
    for _ in range(3):
        project(q, k, v, backend)
    torch.cuda.synchronize()
    times = []
    for _ in range(10):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        project(q, k, v, backend)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def main():
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError('requires ROCm PyTorch and a visible GPU')
    torch.manual_seed(42)
    torch.backends.cuda.preferred_rocm_fa_library('aotriton')
    print(json.dumps(dict(torch=torch.__version__, hip=torch.version.hip,
                          gpu=torch.cuda.get_device_name(), scale=1.0,
                          preferred_backend=str(torch.backends.cuda.preferred_rocm_fa_library()))), flush=True)
    failed = 0
    with torch.inference_mode():
        for dtype in [torch.bfloat16, torch.float16]:
            for hd, kv_heads in [(256, 8), (512, 2)]:
                for kv_len in [256, 1280]:
                    q = torch.randn(1, 256, 16, hd, dtype=dtype, device='cuda').transpose(1, 2) * 0.1
                    k = torch.randn(1, kv_len, kv_heads, hd, dtype=dtype, device='cuda').transpose(1, 2) * 0.1
                    v = torch.randn_like(k)
                    ref = project(q.float(), k.float(), v.float(), SDPBackend.MATH)
                    for layout in ['transposed', 'cached_kv', 'contiguous']:
                        q_in = q.contiguous() if layout == 'contiguous' else q
                        k_in = k if layout == 'transposed' else k.contiguous()
                        v_in = v if layout == 'transposed' else v.contiguous()
                        row = dict(dtype=str(dtype), head_dim=hd, kv_heads=kv_heads,
                                   q_len=256, kv_len=kv_len, layout=layout,
                                   q_stride=q_in.stride(), k_stride=k_in.stride())
                        try:
                            actual = project(q_in, k_in, v_in, SDPBackend.FLASH_ATTENTION)
                            row['nonfinite'] = (~actual.isfinite()).sum().item()
                            error = (actual.float() - ref).abs().max().item()
                            row['max_abs_error'] = error if actual.isfinite().all().item() else None
                            torch.testing.assert_close(actual.float(), ref, atol=0.003, rtol=0.02)
                            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                                project(q_in, k_in, v_in, SDPBackend.FLASH_ATTENTION)
                                torch.cuda.synchronize()
                            operators = [e.key for e in prof.key_averages() if 'attention' in e.key]
                            if not any('_scaled_dot_product_flash_attention' in name for name in operators):
                                raise RuntimeError(f'flash operator absent: {operators}')
                            row.update(status='pass', operators=operators,
                                       flash_ms=time_ms(q_in, k_in, v_in, SDPBackend.FLASH_ATTENTION),
                                       math_ms=time_ms(q_in, k_in, v_in, SDPBackend.MATH))
                        except (AssertionError, RuntimeError) as error:
                            failed += 1
                            row.update(status='fail', error=str(error))
                        print(json.dumps(row, allow_nan=False), flush=True)
    if failed:
        raise RuntimeError(f'{failed} AOTriton probe cases failed; no fallback was used')


if __name__ == '__main__':
    main()
