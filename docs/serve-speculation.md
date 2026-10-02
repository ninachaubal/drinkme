# Speculative decoding

The torch runtime proposes several tokens, then verifies them with the main
model in a batched forward. It can use a checkpoint's multi-token-prediction
(MTP) head or n-gram lookup in the current context. Both use
[`serving/speculative.py`](../src/drinkme/serving/speculative.py).
MLX does not currently support speculation.

## Configuration

```sh
drinkme serve --model Qwen3-8B --spec ngram
```

`--spec` overrides `DRINKME_SPEC`.

| Mode | With an MTP head | Without a head |
|---|---|---|
| `auto` (default) | MTP | N-gram |
| `off` | Serial decode | Serial decode |
| `mtp` | MTP | Serial decode |
| `ngram` | N-gram; head is not loaded | N-gram |
| `ngram+mtp` | Lookup first; MTP on lookup misses | N-gram |

The Qwen3.8-27B head has about 0.4B parameters. Its default reduced pack uses
approximately 0.6 GiB; `DRINKME_MTP_DIET=0` loads the raw approximately 0.75 GiB
head. It runs a small forward per draft token. N-gram lookup needs no model
weights, but its index consumes host memory.

`--spec off` (`DRINKME_SPEC=off`) is the only off switch, for both
proposers. `DRINKME_MTP_DEPTH=k` pins the MTP draft depth and nothing else.
The `DRINKME_MTP_*` variables apply to the MTP head only; the audit, which
covers both proposers, is `DRINKME_SPEC_AUDIT`.

| N-gram variable | Default | Purpose |
|---|---|---|
| `DRINKME_NGRAM_TOKENS` | 5 | Maximum proposed tokens |
| `DRINKME_NGRAM_MAX` | 4 | Maximum match length |
| `DRINKME_NGRAM_MIN` | 3 | Minimum match length |

## Verification and numerical behavior

A cycle verifies k proposals using a k+1-position main-model forward.
At positive temperature, rejection sampling preserves the target distribution.
For a lookup proposal, q is a point mass: accept draft d with probability p[d],
otherwise sample from p with d removed. Histogram tests in
`tests/test_serving_ngram.py` check the emitted distribution against the serial
sampler across seeded draws.

At temperature zero, a draft is accepted when it matches the corresponding
main-model argmax. Batched and single-token forwards can accumulate sums in
different orders, so near-ties may produce different transcripts.
`tests/spec_agree.py` checks argmax agreement where the reference's top-two
margin is outside that near-tie region. N-gram mode leaves prefill unchanged.

### Sliding-window cache rewind

After a rejected draft, the engine must restore every cache state affected by
verification. Gemma-4 sliding-attention layers have two counters, and a full
ring buffer can discard old rows while writing drafts. Rewind restores both
counters and saves/restores the rows that a cycle may shift out.

`serving/mtp.py` describes the implementation;
`tests/test_serving_gemma_spec.py` covers the ring regimes.
`bench/ngram_agreement_gate.py` compares live-server greedy output with
speculation off and on. Without logits on the wire, transcript agreement is
informational rather than a correctness verdict on a divergent near-tie.

## Performance

A lookup miss performs one normal decode forward plus a dictionary lookup.
A match widens the verification batch, which costs time even if the drafts
are rejected. For M≤8, the compressed multi-column kernel reads each weight
once for the batch.

The minimum match length controls weak proposals. A Qwen3-8B tokenizer probe
measured proposals per emitted token:

| `prompt_lookup_min` | uniform random ids | prose (a README) | agent transcript | code re-edit |
| --- | --- | --- | --- | --- |
| 1 (vLLM's default) | 0.054 | 3.244 | 4.886 | 4.907 |
| 2 | 0.000 | 1.152 | 4.768 | 4.756 |
| 3 (drinkme's default) | 0.000 | 0.457 | 4.683 | 4.562 |
| 4 | 0.000 | 0.241 | 4.620 | 4.411 |


In maintainer timing on Strix Halo (Qwen3-8B), a minimum of three kept chat
roughly neutral and helped agent-style workloads, while a minimum of one lost
about 15% on chat. On Qwen3.8-27B, MTP beat lookup on every workload tested,
which is why `auto` prefers the head. `ngram+mtp` remains an explicit
experiment.

<a id="the-cycles-cost-measured"></a>

## Draft and verification timing

`bench/mtp_cycle_floor.py` times draft and verification steps in the engine's
own loop. It can replace compressed kernels with no-op stubs to estimate host
overhead. The draft-cost coefficient `c` is the time for one draft step divided
by the time for one main-model step. In Leviathan's wall-time factor,
`(1−α^{γ+1}) / ((1−α)(γc+1))`, α is acceptance probability and γ is draft depth.

In maintainer runs (Qwen3.8-27B sip, Strix Halo, depth 4), a draft step cost
about 0.055 of a trunk step, verification at M=5 about 1.31× an M=1 step, and
most of the draft chain's time went to reading the full-vocabulary `lm_head`.
Reading fewer head rows could reduce `c`; `DRINKME_MTP_DRAFT_VOCAB` has no
row-subset kernel for a packed head.

## Adaptive bail and re-arm

MTP draft forwards cost time even when their tokens are rejected. To avoid
sustained overhead, the engine tracks acceptance over the last
`DRINKME_MTP_BAIL_WINDOW` drafted positions (default 64). If acceptance over a
full window falls below `DRINKME_MTP_BAIL_FLOOR` (default 0.08), it switches
the request to serial decoding and logs the rate, window, and threshold.
Break-even acceptance on this denominator is about 5–7% for a four-draft cycle.

Both settings are read once per engine, at its first generation. Unparseable values, a window
below 1, or a floor outside 0..1 produce a stderr warning and use the default.
`DRINKME_MTP_BAIL_FLOOR=0` disables the switch to serial decoding.

The denominator includes every **drafted position**. If a cycle drafts four
tokens and rejects the first, all four count as rejected, including the three
positions verification never reached. A request reporting about 70% acceptance
among decided positions can therefore show about 47% in the rolling window.
A low-acceptance region, such as code within prose, can trigger the switch.
Longer windows average over more positions; lower thresholds allow acceptance
to approach break-even before switching.

`bench/bail_sweep.sh` serves each configuration in turn, and
`bench/bail_sweep_table.py` formats the results. A maintainer sweep on a
Qwen3.8-27B sip pack compared a 32/0.15 window/floor with the 64/0.08
defaults. The former triggered in most ordinary sampled free-text runs; the
latter triggered in none of 40 runs across two prompts. Neither prompt produced sustained low acceptance, so this sweep did not test
that case.

After `DRINKME_MTP_REARM` serial tokens (default 64), the engine clears the
window and tries drafting again. If acceptance falls below the threshold
again, the next serial interval doubles: 64, 128, 256, then a cap of 512.
This limits the cost of retrying on consistently poor drafts while allowing
speculation to resume after a temporary low-acceptance region. If acceptance
stays above the threshold for two windows of drafted positions (128 at the
default size), the interval resets. `DRINKME_MTP_REARM=0` keeps the rest of
the request in serial mode.

Serial intervals use the normal one-token step and retain its hidden state.
On resuming, the engine rebuilds the head's attention history in one batched
pass, without an extra trunk forward. Switching modes only controls whether
to draft; rejection sampling, audit counters, and greedy comparison are
unchanged. Audit counters accumulate across the whole request. The n-gram
proposer does not use this mechanism because a lookup miss already performs
a normal decode step.

The log carries one line per trip (`adaptive bail at +41: … serial for the
next 64 tokens`, the number being the generated position) and per re-arm
(`re-armed at +105 after 64 serial tokens`), and the per-request summary ends
with `bailed to serial N× (last at R% rolling), re-armed M×`. `/metrics`
counts both as `drinkme_mtp_bail_trips_total` and
`drinkme_mtp_rearms_total`.

## Counters and index

The log reports emitted tokens, cycles, accepted/proposed drafts, and lookup
match rate. `/metrics` exposes lifetime totals as
`drinkme_spec_proposed_tokens_total` and `drinkme_spec_accepted_tokens_total`,
which count both proposers. `drinkme_mtp_bail_trips_total` and
`drinkme_mtp_rearms_total` are MTP-only.

[`serving/ngram.py`](../src/drinkme/serving/ngram.py) maintains one dictionary
per match length, mapping each n-gram to its latest position. Each context
position is inserted once, giving amortized constant work per token. Measurements
found 1–2 µs/token and about 518 bytes of host memory per context token at
minimum length one (roughly 130 MiB at 262k context). A minimum of four used
about 33 MiB; the default of three lies between them.
