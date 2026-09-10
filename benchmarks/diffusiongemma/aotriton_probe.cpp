#include <aotriton/flash.h>
#include <cstdio>
#include <stdexcept>
#include <vector>

static void check(hipError_t status) {
  if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}

int main() {
  int failures = 0;
  for (auto dtype : {aotriton::kBFloat16, aotriton::kFloat16}) {
    for (uint64_t dim : {256, 512}) {
      const uint64_t seq = 256, heads = 16, kv_heads = dim == 256 ? 8 : 2;
      const size_t qbytes = heads * seq * dim * 2, kvbytes = kv_heads * seq * dim * 2;
      void *q, *k, *v, *out;
      check(hipMalloc(&q, qbytes));
      check(hipMalloc(&k, kvbytes));
      check(hipMalloc(&v, kvbytes));
      check(hipMalloc(&out, qbytes));
      check(hipMemset(q, 0, qbytes));
      check(hipMemset(k, 0, kvbytes));
      check(hipMemset(v, 0, kvbytes));
      check(hipMemset(out, 0x7f, qbytes));
      using namespace aotriton::v3::flash;
      attn_fwd_params p;
      p.Q = T4(reinterpret_cast<intptr_t>(q), {1, heads, seq, dim}, {heads*seq*dim, seq*dim, dim, 1}, dtype);
      p.K = T4(reinterpret_cast<intptr_t>(k), {1, kv_heads, seq, dim}, {kv_heads*seq*dim, seq*dim, dim, 1}, dtype);
      p.V = T4(reinterpret_cast<intptr_t>(v), {1, kv_heads, seq, dim}, {kv_heads*seq*dim, seq*dim, dim, 1}, dtype);
      p.Out = T4(reinterpret_cast<intptr_t>(out), {1, heads, seq, dim}, {heads*seq*dim, seq*dim, dim, 1}, dtype);
      p.Sm_scale = 1.f;
      p.dropout_p = 0.f;
      p.causal_type = CausalType::None;
      auto status = attn_fwd(p, attn_fwd_params::kVersion, aotriton::Stream{});
      check(hipDeviceSynchronize());
      std::vector<uint16_t> result(qbytes / 2);
      check(hipMemcpy(result.data(), out, qbytes, hipMemcpyDeviceToHost));
      size_t nonzero = 0;
      for (auto value : result) nonzero += (value & 0x7fff) != 0;
      std::printf("dtype=%d head_dim=%lu status=%d (%s) nonzero=%zu/%zu\n",
                  int(dtype), dim, int(status), hipGetErrorString(status), nonzero, result.size());
      failures += status != hipSuccess || nonzero != 0;
      check(hipFree(q)); check(hipFree(k)); check(hipFree(v)); check(hipFree(out));
    }
  }
  return failures ? 1 : 0;
}
