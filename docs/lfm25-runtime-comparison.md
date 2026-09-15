# LFM2.5: Candle versus llama.cpp on Strix Halo

September 15 snapshot: our decode speed is close to the existing local llama.cpp
service. Public short-benchmark results are faster, but use different quantization
and workloads. [Structured results](../benchmarks/lfm25/results/2026-09-15-runtime-comparison.json)
preserve the configurations and individual cases.

## Same machine and checkpoint

Zorak has a Ryzen AI Max+ 395 / Radeon 8060S. The existing llama.cpp service on
`127.0.0.1:2031` runs build `b9820-3fc4e1052`, the same LFM2.5 Q5_K_M file,
99 GPU layers, one slot and a 128,000-token context allocation. Its installed
executable links the HIP backend. No vLLM process was found.

One 64-token warmup preceded four full requests using our existing synthetic
adjudication fixtures. The rendered rubric/schema and input match our renderer;
prompt token counts match at 372–381. Requests explicitly select temperature 0,
repetition penalty 1.05 over full history, a 2048-token output cap, no speculative
decoding and no prompt-cache reuse. This overrides llama.cpp's temperature 0.2
and 64-token repetition-window defaults for these requests only.

| Measurement | Candle/lfm2d, saved same-day run | Local llama.cpp, new run |
|---|---:|---:|
| Median decode, cold requests | 102.81 tokens/s | 108.52 tokens/s |
| Median decode, fully cached input | 102.23 tokens/s | Not measured |
| Median full-prompt processing | 533.90 ms | 147.68 ms |

Local llama.cpp is approximately **5–6% faster per generated token** in this
snapshot. Full-prompt processing takes approximately **3.6 times longer** in
Candle. Our rubric cache already reduces normal new-input prefill to roughly
130–158 ms, so the cold-prefill ratio is not the normal cached-request latency
ratio. An exact complete-input cache hit takes only microseconds before decoding.

These runs were not interleaved or isolated. The llama.cpp sample has one measured
request per case; Candle figures come from the earlier buffer experiment. Context
allocations and engine arithmetic differ. Generated texts/lengths differ: llama.cpp
produced 801–1886 tokens, while Candle's cold runs produced 983–1283. Consequently
these are per-token throughput observations, not matched total-answer timings or
quality equivalence. All four llama.cpp requests stopped at EOS. The existing
service remained running with its configuration unchanged.

## Published measurements on the same APU family

| Source / date | Runtime and quantization | Decode | Prompt processing |
|---|---|---:|---:|
| [Local AI Frontier, July 18](https://localaifrontier.com/benchmarks/lfm25-8b-apu-ryzen-ai-max-395-unified-memory/) | llama.cpp HIP, ROCm 7.2.2, Q4_K_M | 150.16 tokens/s, tg256 | 3661.53 tokens/s, pp512 |
| [Strix Halo Guide, June 11](https://github.com/hogeheer499-commits/strix-halo-guide/blob/main/BENCHMARKS.md) | llama.cpp Vulkan/RADV, Q4_K_M | 171.17 tokens/s, tg128 | 3364 tokens/s, pp512 |

Both are first-party measurements of LFM2.5-8B-A1B on Ryzen AI Max+ 395 / 8060S.
The Vulkan source also records 168.96–176.48 tokens/s across nearby June builds.
These synthetic short-generation results suggest a **rough 1.5–1.7× throughput
target relative to our current application rate**, not an established engine-only
gap: Q4 versus Q5, context, sampling overhead, build and power settings differ.

[Liquid AI reports 146 tokens/s on the Ryzen CPU](https://www.liquid.ai/blog/lfm2-5-8b-a1b).
The article text lacks enough precision/workload detail for a matched comparison.
Its high-throughput GPU chart measures SGLang on an H100 in BF16 with concurrent
requests; it is not a Strix Halo vLLM measurement.

No defensible matching **vLLM + LFM2.5-8B-A1B + gfx1151 + single request** number
was found in this search. Benchmarks of other models or aggregate concurrent
throughput should not set our adjudication latency target.

## What this changes

Keep the basic decode fusions in the [next-work assessment](lfm25-next-optimizations.md).
The serving llama.cpp baseline is within single-digit percentage reach. The public
150–176 tokens/s range warrants a future controlled same-file, same-context,
fixed-token comparison of our engine and current llama.cpp HIP/Vulkan builds.
Full-prompt processing retains a larger observed gap; measure cached-suffix
processing separately before prioritizing it against long-answer decode time.
