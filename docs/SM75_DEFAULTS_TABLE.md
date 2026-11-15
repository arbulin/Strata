# The sm_75 defaults table

`nv_defaults` had a table for sm_86, sm_60 and sm_121 but not for 7.5, so a Turing card ran the
sm_75 series only when the user set `STRATA_SM75_PERF=1`. The series exists because nobody on the
team had an sm_75 card to confirm it. This adds the missing table in the same shape as the others:
it sets `STRATA_SM75_PERF=1`, and `STRATA_SM75_PERF=0` is the off switch for the whole card's
tuning. A variable the user set wins, and every other architecture's table is unchanged
(`nv_defaults_test` pins sm_70 and sm_80 at empty).

## What the series turns on

Measured on an RTX 2080 Ti (sm_75, driver 610.57.04, nvcc 13.0, `-arch=sm_75`) against the kernels
they replace. Four are bit-identical; the output hash or the ids were the same in every shape
tried, seven to ten shapes each, alternating rounds:

| switch | path | gain |
|---|---|---|
| `STRATA_PA_V1_PREFETCH` | prompt attention, int8 KV | 1.72x (median of six rounds 1.716x, range 1.579-1.751x), bit-identical |
| `STRATA_ATTN_V3` | decode attention | 1.40-2.12x, 2.0-2.1x at the 2-8 queries a server runs, bit-identical |
| `STRATA_TOPK_SPLIT` | decode top-k | 1.06-11.54x, growing with `--max-context` (1.06x at a 16K-block window, 11.5x at 262146), same ids |
| `STRATA_SCORES_THREAD` | prompt scorer | 2.95 / 5.66 / 8.23x at three block counts, bit-identical |

`PA_V1_PREFETCH` is the one that makes the others matter: the shared K/V buffer takes a block's
shared memory from 37.9 to 29.2 KB, and on a 64 KB Turing SM that is the difference between one
block per SM and two. The kernel is latency-bound, not tensor-core-bound - removing half of every
MMA saves 11.0%, and replacing the int8->half2 conversion with a passthrough saves nothing.

The table also sets `STRATA_PA_V2T`, so Turing takes the v2 prompt-attention kernel (register
staging instead of cp.async) rather than v1: 1.24-1.33x over four shapes and six alternating
rounds. That one is not bit-exact: v1 and v2 sum in another order, 2.1e-07 relative RMS apart on
the output (worst absolute 4.5e-08 against a 0.021 reference rms). `qsa_prompt_attn_parity` passes
with `FAILURES 0` and puts its error against the FP64 reference slightly LOWER than v1's, 6.15e-06
against 6.99e-06 at ctx 32768. Because of that the engine's one-line report does not describe the
sm_75 table as "exact", unlike the sm_86 and sm_60 tables.

## End to end: not established

A paired ABBA on the running server (Qwen3.8-Flash-Next UD-IQ4_XS, `--parallel 8`, layer split,
ON-OFF-OFF-ON, two restarts, two 36.6k-token prompts with 0 reused tokens per arm):

| arm | prefill tok/s |
|---|---|
| `PA_V2T=1` | 1292.9 / 1294.9 / 1280.1 / 1283.4 (mean 1287.8) |
| `PA_V2T=0` | 1272.6 / 1281.6 (mean 1277.1) |

The means are 0.84% apart, but the ranges overlap (1280.1 under 1281.6) and the two ABBA
transitions disagree in sign (+1.75% and -0.12%). That is noise on this server, not a gain. An
earlier pass of three runs an arm showed +1.6% with the ranges apart; the OFF arm's own values had
drifted down by about 1% between the two sessions, which is what closed the gap.

The switch stays in the table because the kernel is 1.24-1.33x faster and it costs nothing, not
because of an end-to-end win. The kernel-to-whole-run ratio here is about 20:1, which is worth
knowing before reading any of the kernel numbers above as a prediction: `SCORES_THREAD`'s 8.23x
and `TOPK_SPLIT`'s 11.5x are kernel times.

The four bit-identical switches were not A/B'd end to end at all. The server these numbers came
from already set `STRATA_SM75_PERF=1`, so they were active before the change and only `PA_V2T` was
ever isolated through the running engine.

## Candidates rejected on the same card

`STRATA_EXPDEC` is 1.12-1.21x on the gate/up phase at T=3-4 (all ten format pairs, bitwise
identical) but -0.4% end to end, because it covers two of this model's gate/up formats and none of
its down projections; the log shows type 18 and 22 captured and 6, 20, 23 and 42 falling back. The
table in docs/EXPERTDEC.md that puts RTX 2080 Ti gate/up at 0.73-0.98x disagrees with the commit
that introduced the kernel (`-11/-23% at T 1-4`) and with this measurement; that row looks stale.

`STRATA_GDN_CHUNKED` is 4.907 ms against 4.181 ms for the token recurrence at T=4096 on this card
(1.480 against 1.499 at T=1024), so leaving it off on CUDA is right. `STRATA_GDN_TC=1` is already
the default for every Linux CUDA build.

`STRATA_MMVF_ROWS` is 12% slower at T=5 (17.88 against 15.86 us per layer) and 2.7x faster at T=8
(19.71 against 53.00), because the kernel takes 8 rows per block once `n_tok > 4` and a
`--spec 4` window is T=5. It is on in the gfx1151 table only, which does not include NVIDIA.

`STRATA_SCORES_HALVE` and `STRATA_TOPK_SAMPLE` are neutral to mildly positive under
`STRATA_SCORES_THREAD=1` (medians 1.01x and 1.02-1.07x), so they stay on. `STRATA_QSA_ROWS` is
inert on the scorer at n_q 1, 4, 64 and 512. Raising `nv_chunk`'s Turing ceiling does not apply to
a server that passes an explicit `--prefill auto:N`, which already bypasses it.

## Validation status

`nv_defaults_test` passes with no GPU: the table's two entries, `STRATA_SM75_PERF=0` and `=1`,
that `STRATA_DEFAULTS=026` does not touch it, that no other architecture's table changed, and that
sm_70 and sm_80 stay empty. The full engine builds for sm_75 with 0 errors. `qsa_prompt_attn_parity`
passes with `FAILURES 0` with the series on. HIP, SYCL and Metal were not validated and are not
touched: this table is NVIDIA-only and the gates it relies on are the ones `sm75_default` already
used.
