"""Compile and validate the native gfx1151 canvas-attention kernel bundle."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.kernel_dir.resolve()))
    from aotriton_canvas_kernel import canvas
    from fwd_kernel_inner import IS_JIT_COMPILING
    if not IS_JIT_COMPILING:
        raise RuntimeError('enable IS_JIT_COMPILING in the exported upstream kernel')
    target = triton.runtime.driver.active.get_current_target()
    if target != GPUTarget('hip', 'gfx1151', 32):
        raise RuntimeError(f'expected gfx1151, got {target}')
    if triton.__version__ != '3.5.1':
        raise RuntimeError('native ABI is validated with Triton 3.5.1 only')
    torch.manual_seed(42)
    manifest = dict(abi=1, arch='gfx1151', dtype='f16', max_q=256, max_kv=8192,
                    triton=triton.__version__, upstream='6e00ef3e335b45dfb49065259533b59c68995bfe',
                    kernels=[])
    source_hash = hashlib.sha256()
    for p in sorted(args.kernel_dir.glob('*.py')):
        source_hash.update(p.name.encode()); source_hash.update(p.read_bytes())
    manifest['upstream_source_sha256'] = source_hash.hexdigest()
    if source_hash.hexdigest() != '2891bf8ac09d463a2ac34852d992a0ea5d58731312a81d8bc1057fac47b64876':
        raise RuntimeError('expected pinned upstream source with only IS_JIT_COMPILING enabled')
    manifest['wrapper_sha256'] = hashlib.sha256(Path(canvas.fn.__code__.co_filename).read_bytes()).hexdigest()
    signature = {name: '*fp16' for name in ['Q', 'K', 'V', 'Out']}
    signature.update(L='*fp32', Q_LEN='i32', KV_LEN='i32')
    signature = {name: signature[name] for name in ['Q', 'K', 'V', 'L', 'Out', 'Q_LEN', 'KV_LEN']}
    for d, hk in [(256, 8), (512, 2)]:
        compiled = triton.compile(ASTSource(canvas, signature, constexprs={'D': d, 'HK': hk}),
                                 target=target, options={'num_warps': 8, 'num_stages': 1, 'waves_per_eu': 1})
        # The native launcher supplies seven arguments plus Triton's two unused scratch pointers.
        declaration = next(x for x in compiled.asm['llir'].splitlines()
                           if 'define amdgpu_kernel void @canvas(' in x)
        if declaration.count('ptr addrspace') != 7 or declaration.count('i32 inreg') != 2:
            raise RuntimeError(f'unexpected native ABI: {declaration}')
        if getattr(compiled.metadata, 'global_scratch_size', 0) or compiled.metadata.profile_scratch_size:
            raise RuntimeError('native ABI does not allocate Triton scratch buffers')
        checks = []
        for qlen, kvlen in [(256, 256), (256, 1280), (253, 1277), (1, 17), (256, 8192)]:
            for magnitude in [0.1, 1.0]:
                q = torch.randn(1, 16, qlen, d, device='cuda', dtype=torch.float16) * magnitude
                k = torch.randn(1, hk, kvlen, d, device='cuda', dtype=torch.float16) * magnitude
                v = torch.randn_like(k)
                out = torch.full_like(q, float('nan'))
                lse = torch.empty((16, qlen), device='cuda', dtype=torch.float32)
                compiled[(triton.cdiv(qlen, 32), 16, 1)](q, k, v, lse, out, qlen, kvlen)
                torch.cuda.synchronize()
                with sdpa_kernel(SDPBackend.MATH):
                    ref = torch.nn.functional.scaled_dot_product_attention(
                        q.float(), k.float(), v.float(), scale=1., enable_gqa=True)
                torch.testing.assert_close(out.float(), ref, atol=.003, rtol=.02)
                row = dict(q=qlen, kv=kvlen, magnitude=magnitude,
                           max_abs_error=(out.float()-ref).abs().max().item())
                checks.append(row)
                print(json.dumps(dict(head=d, **row)), flush=True)
        binary = compiled.asm['hsaco']
        file = f'canvas-d{d}.hsaco'
        (args.output / file).write_bytes(binary)
        (args.output / f'canvas-d{d}.amdgcn').write_text(compiled.asm['amdgcn'])
        (args.output / f'canvas-d{d}.llir').write_text(compiled.asm['llir'])
        manifest['kernels'].append(dict(head_dim=d, kv_heads=hk, file=file, name=compiled.metadata.name,
                                      sha256=hashlib.sha256(binary).hexdigest(), block_m=32,
                                      threads=256, shared_bytes=compiled.metadata.shared, checks=checks))
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('PASS: validated kernel bundle published', flush=True)


if __name__ == '__main__':
    main()
