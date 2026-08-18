# Contributing

This is a **reference implementation** for the method in
[arXiv:2606.05145](https://arxiv.org/abs/2606.05145), maintained alongside active research. It is
meant to be readable and reusable rather than to grow features quickly.

**Issues are welcome**: bug reports, unclear docs, a model family that does not load, a result you
cannot reproduce. These are the most useful contributions.

**Please open an issue before a pull request.** The operators and features here back published
numbers, and a change that looks like a cleanup can silently change a result. Agreeing on what should
change first saves you the wasted work.

## What is easy to accept

- **A new model family.** Subclass the dual backbone in `backbones.py`, add a `register_*` line in
  `register.py`, add the `model_type → register` entry in `pipeline._ARCH_REGISTER`. Keep the
  dot-anchored `stacked_params_mapping`, which is load-bearing for correct weight loading.
- **Docs and error messages**, especially anywhere the failure mode was not obvious to you.
- **Tests**, particularly Tier-0 ones that need no GPU.

## What needs discussion first

- Changing an operator's semantics, the junction rule, the feature definitions, or the routing rule.
  These define what the numbers mean.
- New dependencies. See the boundary below.
- Anything that moves work from the caller into the library (config handling, path conventions,
  cluster or scheduler awareness). That coupling is what the boundary check exists to prevent.

## Ground rules

**Three dependency tiers, and the first one is load-bearing.**

| Tier | Needs | Contains |
|---|---|---|
| 0 | numpy | `routing`, `io`, `demo`. Must stay importable on a laptop |
| 1 | + torch | `storage`, `junctions`, `alpha`, `scoring` |
| 2 | + vLLM | `pipeline`, `features`, `repair`, `processors`, `backbones` |

Tier 1 and 2 symbols are imported lazily in `__init__.py`, so `import vllm_logits` never pulls torch
or vLLM. `python tools/check_import_boundary.py` enforces this and runs in CI. If it fails, the fix is
usually to move a symbol behind the lazy loader rather than to widen the allowlist.

**Every vLLM-internal import lives in `_compat.py`.** The dual backbones subclass vLLM internals, so a
vLLM release can break them. Keeping those imports in one file makes a version bump a single-file
diff. Do not import vLLM internals anywhere else.

**Before submitting**

```bash
python tools/check_import_boundary.py
pytest tests/test_routing.py tests/test_demo.py -q     # Tier 0, no GPU
ruff check src tools tests
```

If you touched the engine path and have a GPU, also run the dual-load tests. They confirm the two
checkpoints load and reproduce exact greedy outputs:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1
pytest tests/test_dual_load_qwen.py tests/test_dual_load_phi4.py -s
```

If you do not have a GPU, say so in the PR. That is fine, it just tells us what to check.
