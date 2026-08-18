# Architecture

Five things here are harder than they look. This page is about those five: what each one solves, and
what breaks if you change it. Read it before a pull request, and when a vLLM upgrade turns something
red.

```
      specialist weights ─┐
                          ├─▶ ONE vLLM model, both backbones resident
      ancestor weights  ──┘   both hidden states available before the LM head
                                              │
                          logits processor combines them, in the sampling hot path
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
        [1] cache_logits                                        [2] repair_logits
     two batched prefills →                              operator fires at a chosen
     per-token divergence →                              position → re-decode →
     junction = where it peaked                          correct at k?
                    │                                                   │
                    └──────────────▶ sharded feature store ◀────────────┘
```

---

## 1. Two checkpoints in one vLLM model

**The problem.** Comparing a fine-tune against its base normally means two models, two forward passes,
and two KV caches. Inside a sampling loop that is the whole latency budget.

**What we do.** `backbones.py` defines `DualQwen2ForCausalLM` (and Llama / Phi3 siblings): a single
vLLM model that loads *both* checkpoints' backbones and exposes both final hidden states before the
LM head. A logits processor combines them with no extra forward pass.

**Why mixing at the final hidden state is legitimate.** `lm_head` is a pure linear projection, so
mixing hidden states and mixing logits are the same operation. That identity is what makes the
zero-overhead claim true rather than an approximation.

**The KV-cache trap.** vLLM assigns KV cache slots by parsing layer indices out of module names
(`extract_layer_index`). Two backbones in one model produce two layers numbered `0`, which collide.
`_ShiftedLayerList` registers the second backbone's children under shifted integer keys
(`VLLM_LOGIT_MIX_A_LAYERS` sets the offset) so every layer gets a unique slot. Remove the shifting and
you get silent cache aliasing. Not a crash, wrong outputs.

**The weight-loading trap. This is the sharpest edge in the repo.** `stacked_params_mapping` is
**dot-anchored** (`".v_proj"`, not `"v_proj"`) because the loader rewrites parameter names by
substring replacement, and `"v_proj"` is a substring of `"qkv_proj"`. Un-anchored, loading a model
with pre-stacked QKV weights (Phi-3) silently corrupts them. This looks exactly like a redundant dot
that a tidy-up would remove. Do not remove it. `tests/test_dual_load_{qwen,phi4}.py` exist to catch
it. They assert the dual model reproduces *exact greedy outputs*, which is the only assertion that
detects a subtly wrong weight load.

**Cost, stated plainly.** Both checkpoints are resident in GPU memory, so the footprint is roughly
two models. That is the real limit on model size per GPU, and offloading the second stream is the
main open piece of work.

## 2. Registration that survives process forks

**The problem.** vLLM spawns `EngineCore` worker processes. A model registered in your main process
does not exist in the workers, so a custom architecture fails at load with a confusing registry error.

**What we do.** The package declares a `vllm.general_plugins` entry point
(`vllm_logits.register:register_all`). vLLM calls entry points in *every* process it creates, so the
dual backbones register in the workers too, and users never call `register_*()` at all.

**Consequence to know.** Because registration happens at plugin load and the backbones read their
configuration from environment variables, `VLLM_LOGIT_MIX_*` must be set **before** `LLM()` is
constructed. `pipeline.py` and `repair.py` do this for you; if you drive the backbones directly, set
them first or the workers will fork with the wrong configuration.

## 3. Feature extraction: two prefills, not a decode loop

**The problem.** The per-token divergence between specialist and ancestor is needed over a *finished*
trace. Re-generating it token by token would cost a full decode per rollout.

**What we do.** `features.py` runs two batched prefills over the completed text, one per model, and
computes every per-token quantity from the resulting logits: divergence on the taken token, coverage
divergence over the ancestor's top-k set, logit variance, entropy, local KL, and the combined
divergence signal whose peak defines the **junction**. Prefill is parallel over positions, so this is
a small multiple of one forward pass rather than a generation.

**Definitions.** [`docs/logit-features.md`](docs/logit-features.md) gives the formula for each
feature, and is the source of truth over any prose here.

## 4. The logits processor is in the hot path

**The problem.** Everything in `processors/` runs inside sampling, per token, per sequence. It has to
be cheap and it has to be exactly right at the position it fires.

**What we do.** Each processor is a closure created per request from `SamplingParams.extra_args`, so
the per-token work is an index comparison and, at the firing position, one tensor combine. Ancestor
logits for an intervention window are pre-loaded before decoding starts, so a firing position never
triggers disk I/O mid-generation.

**`continuation_mode` is a scientific control, not a knob.** After an intervention fires, the rest of
the sequence is decoded either by sampling the specialist at `T` (`temperature`, the realistic
number, since the retry baseline decodes the same way) or greedily (`greedy`, which isolates the
intervention so a rescue is attributable to the steer rather than to lucky downstream sampling). Which
one you choose changes what the resulting number *means*. Pick deliberately and report which.

## 5. Two firewalls that keep this maintainable

**`_compat.py`, the vLLM version firewall.** The dual backbones subclass vLLM internals, which are
not a stable API. Every single vLLM-internal import in the package is confined to `_compat.py`, so a
breaking release is a one-file diff instead of a hunt. The support policy follows from this: run
`tests/test_dual_load_*.py` after a vLLM upgrade; if they are green, that version works. Currently
pinned to `vllm>=0.15,<0.16`.

**`tools/check_import_boundary.py`, the coupling firewall.** This library must stay embeddable, which
means it may not learn about config frameworks, experiment harnesses, schedulers, or anyone's
directory layout. The check walks every module's AST and fails on imports outside an allowlist, on
private-harness environment variables, and on Tier-0 modules importing torch or vLLM. It runs in CI.
It exists because that property is easy to state and easy to lose to one convenient import.

### The three tiers, and why the first one matters

| Tier | Needs | Modules | Why |
|---|---|---|---|
| 0 | numpy | `routing`, `io`, `demo` | Routing failures, reading results and running the worked example need no model. Keeping this tier honest means someone can try the method on a laptop in under a minute. |
| 1 | + torch | `storage`, `junctions`, `alpha`, `scoring` | Tensor work on cached features, still no vLLM and no GPU required. |
| 2 | + vLLM | `pipeline`, `features`, `repair`, `processors`, `backbones` | Anything that loads a model. |

`__init__.py` imports Tier 1 and 2 lazily through `__getattr__`, so `import vllm_logits` never pulls
torch or vLLM, and a missing extra produces a message naming the install command rather than a
traceback. CI asserts both properties on every push.

## Where to make a change

| You want to… | Go to | Watch out for |
|---|---|---|
| add a model family | `backbones.py` subclass → `register.py` line → `pipeline._ARCH_REGISTER` entry | keep `stacked_params_mapping` dot-anchored; specialist and ancestor must share an architecture |
| change how an operator steers | `processors/logit_repair.py` | it is in the hot path, and it changes what published numbers mean, so open an issue first |
| add a per-token feature | `features.py` + `docs/logit-features.md` | old caches will not have it; `storage.py` auto-detects format but not schema |
| change junction detection | `junctions.py` | the junction defines where every targeted operator fires |
| route differently | `routing.py` (Tier 0, pure numpy) | supply a `RoutingPolicy` instead of editing the default if you only need your own mapping |
| survive a vLLM bump | `_compat.py`, then `tests/test_dual_load_*.py` | nothing else should import vLLM internals |
