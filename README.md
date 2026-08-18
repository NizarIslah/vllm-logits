# vllm-logits

[![CI](https://github.com/NizarIslah/vllm-logits/actions/workflows/ci.yml/badge.svg)](https://github.com/NizarIslah/vllm-logits/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2606.05145-b31b1b.svg)](https://arxiv.org/abs/2606.05145)

### Your fine-tune failed a problem. Should you spend more samples, or is it structural?

Sampling again is the default answer, and it is often the wrong one. Some failures are unlucky
draws. Others are a trajectory the model will re-derive at any temperature, so more rollouts buy the
same wrong answer again. `vllm-logits` reads which is which off the failed generation itself, in one
pass, and names the intervention that fixes the recoverable ones.

It loads your fine-tune (the *specialist*) and the model it was trained from (the *ancestor*) into a
single vLLM model, so comparing them costs **no extra forward pass**.

```bash
pip install git+https://github.com/NizarIslah/vllm-logits
python -m vllm_logits.demo        # 1,423 real failed problems, no GPU, about a minute
```

Two stages:

1. **extract per-token features** from failed generations, measuring how far the specialist drifted
   from the ancestor at each step, and where the failure was decided.
2. **repair** those failures by steering decoding back toward the alternative the ancestor still
   ranks highly, then report which failures were fixable and by which operator.

Inputs are plain: two model paths, your prompts, your rollouts, and a one-line correctness function.
It runs on any model and any task, on a cluster or a single GPU.

```python
from vllm_logits import LogitPipeline, numeric_answer

pipe = LogitPipeline(
    specialist="Qwen/Qwen3-0.6B",        # any HF id or local path
    ancestor="Qwen/Qwen3-0.6B-Base",     # base / stronger / cross-family reference
    arch="auto",                          # detects the right dual backbone from the model config
    T=0.6, alpha=0.7,                     # decode temperature; mixing strength
    continuation_mode="temperature",      # how to decode after an intervention (see below)
)

# Inputs are plain dicts (or JSONL). No framework objects, no registries:
#   problem = {"problem_id": str, "prompt": str, "answer": ...}
#   rollout = {"problem_id": str, "rollout_idx": int, "generated_text": str, "is_correct"?: bool}

results = pipe.run(
    problems, rollouts,
    correctness_fn=numeric_answer("answer"),   # the only task-specific input
    operators=["geo", "rand", "dense", "local_temp"],
    k_values=[1, 5, 10],
)
# results[i] = {"pid": ..., "retry": {...}, "geo_pred": {...}, "rand": {...}, "dense": {...}, ...}
# pipe.run() = extract features over the failed rollouts, then sweep the repair operators.
# You can also call the two stages separately: pipe.cache_logits(...) then pipe.repair(...).
```

## The demo, in more detail

```bash
python -m vllm_logits.demo                      # all cells
python -m vllm_logits.demo --cell sft0p6b\|gsm8k  # one model and task
python -m vllm_logits.demo --plot panel.png     # also write the figure
```

The core install is numpy and nothing else. The demo runs on 1,423 real failed problems shipped with
the package, from four post-trained models across three tasks (provenance:
[`src/vllm_logits/data/README.md`](src/vllm_logits/data/README.md)). It prints how many failures are
worth more sampling, how many need a different intervention, how many are beyond reach, and whether
routing each failure by its features beats committing to one intervention everywhere.

## Install

```bash
pip install git+https://github.com/NizarIslah/vllm-logits              # core: numpy only
pip install "git+https://github.com/NizarIslah/vllm-logits#egg=vllm-logits[engine]"   # + vLLM/torch
```

| install | needs | gives you |
|---|---|---|
| core | numpy | `route()` on your own features, the input contract, `python -m vllm_logits.demo` |
| `[engine]` | + vLLM 0.15.x, torch, transformers | `LogitPipeline`: loading models, extracting features, running operators |
| `[demo]` | + matplotlib | the demo's figure |
| `[dev]` | + pytest | the test suite |

Requires **Python 3.10 or newer**. The engine extra pins **`vllm` 0.15.x** (see
[Compatibility](#compatibility)). `import vllm_logits` never imports torch or vLLM, so the core
install stays light. Asking for an engine symbol without the extra raises an error naming the
install command. Not on PyPI yet.

## How it works

```
  specialist (your fine-tune)  ─┐
                                ├──▶  ONE vLLM model: both logit streams,
  ancestor   (base / reference)─┘     mixed/steered before the LM head
                                      (no extra forward pass)
                                              │
   problems + failed rollouts + correctness_fn
                                              │
                                              ▼
  [1] cache_logits  ── at each token of a failed trace, measure how far the
          fine-tune drifted from its base model, and find the junction (if any):
          the window of tokens where the failure was decided
                                              │
                                              ▼
  [2] repair  ── at the junction, apply an operator and re-decode
          retry · sparse steer · dense steer · local temperature
                                              │
                                              ▼
   ▶  which failures are fixable, and by which operator

   ...and the 3 trace features route each failure with no repair outcomes:
          spread            ──▶  dense steer
          concentration     ──▶  sparse steer
          logit dispersion  ──▶  local temperature lift
```

- **Two backbones in one model.** `backbones.py` defines `DualQwen2/Llama/Phi3ForCausalLM`: a single
  vLLM model that loads both checkpoints and exposes both logit streams before the LM head, so a
  logits processor can combine them with zero extra passes. `arch="auto"` picks the right one from
  the model's config. Adding a new architecture is one subclass plus one registration line.
- **Logits processors** (`processors/`): `inject` (target chosen positions), `proxy_tuning`
  (specialist + α·(specialist − ancestor) style arithmetic, a reference implementation of
  [proxy-tuning](https://arxiv.org/abs/2401.08565), which is similar in spirit to the logit
  steering here), `cross_arch` (mismatched tokenizers), and `logit_repair` (the steering processor used
  by the repair stage).
- **Feature extraction** (`features.py`, the `cache_logits` stage): for each failed rollout it runs
  two batched prefills and stores per-token features. Those are the specialist to ancestor
  divergence on the taken token (`Delta_path`), coverage-set divergence (`G_cov`), logit variance,
  entropy, local KL, and a combined `J_approx`. Every feature is defined in
  [docs/logit-features.md](docs/logit-features.md). The **junction** is the window of tokens around
  the `J_approx` peak where the failure was decided, and it is where the targeted operators act.
- **Repair** (`repair.py`, the `repair_logits` stage): it applies each operator and re-decodes,
  then reports whether the result is correct at each `k`. Sparse logit steering and local temperature
  lift fire **at the junction**, random-position steer fires at a control position, and dense steer
  applies over the whole trace.

### `continuation_mode`

After an intervention fires, the rest of the sequence is decoded one of two ways, applied uniformly
to every operator:

| mode | post-intervention decode | use it for |
|---|---|---|
| `temperature` *(default)* | sample the specialist at `T` (same as the retry baseline) | **production**, the realistic recoverability number |
| `greedy` | argmax of the specialist | **analysis**, which isolates the intervention so a rescue is attributable to the steer rather than to lucky downstream sampling |

## How this differs from other logit-space methods

| | What it does | Relationship to this |
|---|---|---|
| **Best-of-N, self-consistency** | draw more samples from the same distribution | This decides whether that will work before you pay for it. Complementary: the answer is often "yes, resample". |
| **Proxy tuning** | steer a large model using the delta between a tuned and untuned small pair | Same operator class, and shipped here as a reference implementation (`processors/proxy_tuning.py`). The difference is that this is diagnostic first: it localizes where to steer, and whether steering is the right move at all. |
| **DoLa, contrastive decoding** | contrast layers or model sizes to improve factuality, applied uniformly | Uniform application, no diagnostic for which failures to apply it to. Here the contrast is against a separate ancestor checkpoint and fires at one detected position. |
| **Speculative decoding** | two models for throughput, outputs unchanged | Two models for diagnosis, outputs deliberately changed. |

The distinction that matters: those methods change generation. This one first decides whether
changing generation can help, then picks the change. It is not a claim to beat them. There is no
head-to-head comparison here, and the paper lists that as a limitation.

## Worked example 1: three outcomes of a failure, explained by features

`examples/showcase_three_regimes.py` runs on **Qwen3-0.6B** (specialist) against
**Qwen3-0.6B-Base** (ancestor) over a small set of simple arithmetic problems, for example
`Compute 17 * 23. Put the final answer in \boxed{}.`. It takes the problems the specialist fails on
every sampled rollout and classifies what, if anything, rescues each, alongside the junction-feature
profile. **The output below is real**, regenerated by the script:

```
PROBLEM    retry  rand   geoP   geoW   dense  Ltemp    V_traj  V_junc   kl_jc  Gcov_jc  Dpath_jc
p5         -      OK     OK     -      -      OK        0.151   0.143    1.06    0.195     1.231   [STEERABLE]
p11        -      -      -      -      -      OK        0.102   0.181   11.50    0.032     1.586   [STEERABLE]
p2         OK     OK     -      OK     -      OK        0.161   0.288   12.37    0.087     1.963   [SAMPLING]
p3         OK     OK     OK     -      -      OK        0.154   0.268    8.75    0.121     2.105   [SAMPLING]
p9         -      -      -      -      -      -         0.108   0.222    4.16    0.018     1.948   [HARD]   739*856
p14        -      -      -      -      -      -         0.119   0.196    5.60    0.019     1.869   [HARD]   12345*6789
regime counts: {'SAMPLING': 7, 'STEERABLE': 3, 'HARD': 4}
```

Columns are the operators, where `OK` means rescued at some `k`: `retry` (resample), `rand`
(ancestor injection at a random position), `geoP` and `geoW` (sparse steer at the detected junction
versus a control position), `dense` (steer at every position), `Ltemp` (local temperature lift). The
right-hand columns are the junction-feature profile (`V_traj`, `V_junc`, KL, `G_cov`, `Delta_path` at
the junction).

- **SAMPLING**: plain resampling fixes it. The first failure was an unlucky draw.
- **STEERABLE**: resampling fails, but a logit intervention fixes it. The correct alternative was
  present but suppressed, and steering toward the ancestor surfaces it.
- **HARD**: nothing fixes it. Notice `Gcov_jc` collapses to about 0.02 on the hard multiplications.
  The ancestor is *also* wrong there, so there is no correct alternative to steer toward. That
  collapse is the readable signature of a genuinely unrecoverable failure.

## Worked example 2: routing failures to a repair operator from features alone

`examples/showcase_clustering.py` reduces each failed problem to **three trajectory features**. Each
feature maps to the one operator it makes actionable:

| feature | definition | routes to |
|---|---|---|
| **spread** | `J_frac+`, the fraction of trace tokens with `J_approx > 0` (how broad the divergence is) | `dense steer` |
| **concentration** | `log10(J_max / J_mean)`: one sharp spike versus diffuse | `sparse steer` |
| **logit dispersion** (temperature sensitivity) | `log10(V_t*)`, the variance of the specialist's logits at the junction: how strongly the token responds to a temperature change | `local temperature lift` |

The **prospective routing rule** z-scores the three features per problem and routes each failure to
the operator whose feature is largest (`argmax`, no gate). It reads the failed trace alone, with no
repair outcomes needed.

A **single panel** carries both signals on every point, over a 2-D projection of the three features:
the **color** is the operator the feature rule picks, and the **label** is the failure's empirical
recoverability at a best-of-3 budget. `retry-solvable` means plain resampling fixes it and no routing
is needed. `hard` means it does not, so a logit intervention or a local temperature lift is what can
still help, which is the paper's routing target. In Example 1's terms, `retry-solvable` is the
SAMPLING regime and `hard` is STEERABLE and HARD combined.

![Each failure: routed operator (color) and retry-solvable vs. hard (label)](docs/clustering.png)

```
routing features (z-scored):  spread -> dense steer | concentration -> sparse steer | logit dispersion -> local temperature lift
--------------------------------------------------------------------------------------------
routed operator            n            spread     concentration  logit dispersion
sparse steer               7             -0.43              0.88             -0.20
dense steer                7              0.60             -0.73             -0.65
local temperature lift     6             -0.20             -0.17              0.99
--------------------------------------------------------------------------------------------
recoverability (empirical): {'retry-solvable': 13, 'hard': 7}
over 20 failing problems
```

Each routed bucket has its own signature feature highest. `dense steer` has the highest `spread`,
`sparse steer` the highest `concentration`, and `local temperature lift` the highest
`logit dispersion`, so the rule reads the same way on your own data.

Scope: this runnable example uses a 0.6B pair on arithmetic so that it fits on one GPU, which makes
the split illustrative rather than a result. The shipped demo data
(`python -m vllm_logits.demo`) is the real thing: 1,423 failed problems from post-trained 0.6B to 4B
models on GSM8K, CruxEval and GPQA. Results here are cached to `docs/clustering_data.json` so
re-plotting needs no GPU. Delete it or set `VLLM_LOGITS_FORCE=1` to recompute.

## API reference

| Module | Contents |
|---|---|
| `pipeline.py` | `LogitPipeline`, the one entry point (`.run`, `.cache_logits`, `.repair`). |
| `backbones.py` | `DualQwen2/Llama/Phi3ForCausalLM` plus a repair variant: two checkpoints in one vLLM model, mixable logits. |
| `processors/` | `inject`, `proxy_tuning`, `cross_arch` and `logit_repair`, the steering processor with `continuation_mode`. |
| `register.py` | `register_dual_{qwen,llama,phi3}`, vLLM ModelRegistry shims, auto-loaded (see below). |
| `features.py` | `cache_logits`: per-token features (`Delta_path`, `G_cov`, `logit_var`, `entropy`, `kl_div`, `J_approx`, defined in [docs/logit-features.md](docs/logit-features.md)) via two batched prefills. |
| `scoring.py` | `compute_scores_at_t`, `estimate_background`, and reference (HF) engines. |
| `junctions.py` | junction detectors (Page-Hinkley, divergence, coverage, random) plus offline firing on cached features. |
| `repair.py` | `LogitRepairEngine`: batched operator by `k` sweep. |
| `storage.py` | `LogitStore`, feature cache as `.pt` or parquet shards, format auto-detected on read. |
| `io.py` | the input contract: `Problem` and `Rollout` dicts, JSONL helpers, and the `exact_match`, `numeric_answer` and `regex` correctness defaults. |
| `routing.py` | `route`, `route_scores`, `RoutingPolicy`: the feature to operator rule, pure numpy, no vLLM and no GPU. |
| `demo.py` | `python -m vllm_logits.demo`, the worked example on shipped data. |
| `alpha.py` | `entropy_gap`, `chi2_divergence`, `adaptive_alpha`. |
| `_compat.py` | every vLLM-internal import, isolated in one place: the version-bump firewall. |

Registration is automatic. The package declares a `vllm.general_plugins` entry point, so the dual
backbones are registered in every vLLM process, including spawned workers, without you calling
`register_*()`.

## Tests

```bash
pytest tests/test_routing.py tests/test_demo.py     # no GPU, no torch, no vLLM
python tools/check_import_boundary.py               # the dependency boundary (also in CI)
PYTHONPATH=src python -m pytest tests/test_cache_logits_parquet.py        # CPU: cache round-trip
# GPU: confirm the dual backbones load and reproduce exact greedy outputs:
export VLLM_WORKER_MULTIPROC_METHOD=spawn NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1
PYTHONPATH=src python -m pytest tests/test_dual_load_qwen.py tests/test_dual_load_phi4.py -s
```

## Supported model families

`arch="auto"` reads the model's `config.model_type` and selects the matching dual backbone. The
specialist and ancestor must share an architecture, since they are loaded into one vLLM model.
Supported families:

| `model_type` | dual backbone | example models |
|---|---|---|
| `qwen2`, `qwen3` | `DualQwen2ForCausalLM` | Qwen3-0.6B/1.7B/4B/8B, Qwen2.5 / Qwen2.5-Math (incl. R1-Distill-Qwen) |
| `phi3` | `DualPhi3ForCausalLM` | Phi-3, Phi-4-mini (instruct / reasoning) |
| `llama` | `DualLlamaForCausalLM` | Llama-family checkpoints |
| `olmo2`, `olmo3` | `DualQwen2ForCausalLM` | OLMo-2 / OLMo-3 (Qwen2-shaped) |

Qwen and Phi are verified end to end on GPU. Llama and OLMo run through the same backbone code path.
Anything not listed takes a one-time addition, described below.

See [ARCHITECTURE.md](ARCHITECTURE.md) for how the dual load, the worker-safe registration, the
feature extraction and the two firewalls actually work, and
[CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## Adding a new architecture

Subclass `DualQwen2ForCausalLM` in `backbones.py`, add a `register_*` line in `register.py`, and add
the `model_type` to `register` entry in `pipeline._ARCH_REGISTER`. The dot-anchored
`stacked_params_mapping` in `backbones.py` is load-bearing for correct weight loading, so keep it.

## Compatibility

Pinned to **`vllm>=0.15,<0.16`**, tested on vLLM 0.15.1 with torch 2.9.1 on NVIDIA L40S, A100 and
H100. The dual backbones subclass vLLM internals, all confined to `_compat.py`. Run the dual-load
tests after a vLLM upgrade. If they stay green, the version is supported.

## Citation

If you use `vllm-logits`, please cite the accompanying paper:

```bibtex
@misc{islah2026failedreasoningtraces,
  title         = {Failed Reasoning Traces Tell You What Is Fixable (But Not by Reading Them)},
  author        = {Islah, Nizar and others},
  year          = {2026},
  eprint        = {2606.05145},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2606.05145},
}
```

## License

Apache-2.0.
