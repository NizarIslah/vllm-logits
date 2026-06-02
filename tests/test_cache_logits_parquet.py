"""Round-trip verification for the parquet feature-cache format vs legacy .pt.

Now imports
the cell-feature helpers from `vllm_logits.features` (severed from the repo's
analysis module). CPU-only; no real artifacts.

Checks, on synthetic payloads, that:
  [1] save_feature_shard -> iter_feature_rollouts preserves tensors,
  [2] load_feature_cache(pid,ridx) reads from parquet shards (the repair read-back path),
  [3] the analysis reader (_load_cell_rollouts -> _rollout_feats) yields IDENTICAL
      junction/trace features for a parquet cell and an equivalent .pt cell.

Run: python -m pytest tests/test_cache_logits_parquet.py   (needs polars)
"""
from __future__ import annotations
import tempfile

import torch

from vllm_logits.storage import LogitStore
from vllm_logits.features import (
    _load_cell_rollouts, _rollout_feats, _CELL_CACHE, FEATS)


def _payloads(n=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        T = int(torch.randint(20, 60, (1,), generator=g).item())
        p = {"pid": f"sample_{i}", "rollout_idx": 0}
        for f in FEATS:
            p[f] = torch.rand(T, generator=g)
        # kl_div with some leading-zeros (mirrors real caches; _rollout_feats trims by it)
        kl = torch.rand(T, generator=g); kl[: T // 5] = 0.0
        p["kl_div"] = kl
        out.append(p)
    return out


def _run(w=16, tol=1e-5):
    payloads = _payloads()
    with tempfile.TemporaryDirectory() as tmp:
        store = LogitStore(tmp)
        task = "demo"
        # --- parquet cell ---
        store.save_feature_shard(task, "pqcell", payloads[:4], 0)
        store.save_feature_shard(task, "pqcell", payloads[4:], 1)
        pq_dir = str(store.feature_cache_dir(task, "pqcell"))
        # [1] iter round-trip count
        n_iter = sum(1 for _ in store.iter_feature_rollouts(task, "pqcell"))
        assert n_iter == len(payloads), f"iter count {n_iter} != {len(payloads)}"
        # [2] load_feature_cache from parquet
        got = store.load_feature_cache(task, "pqcell", "sample_3", 0)
        assert got is not None and "logit_var" in got, "load_feature_cache parquet fallback failed"
        assert torch.allclose(got["logit_var"], payloads[3]["logit_var"], atol=tol), "parquet tensor mismatch"
        # [3] .pt cell with the SAME payloads
        for p in payloads:
            store.save_feature_cache(task, "ptcell", p["pid"], 0, dict(p))
        pt_dir = str(store.feature_cache_dir(task, "ptcell"))

        _CELL_CACHE.clear()
        pq_feats = {pid: _rollout_feats(d[0], w) for pid, d in _load_cell_rollouts(pq_dir).items()}
        pt_feats = {pid: _rollout_feats(d[0], w) for pid, d in _load_cell_rollouts(pt_dir).items()}
        assert set(pq_feats) == set(pt_feats), f"pid sets differ: {set(pq_feats)^set(pt_feats)}"
        max_err = 0.0
        for pid in pq_feats:
            for k in pq_feats[pid]:
                # .pt stores float16 -> small rounding vs parquet float32; allow loose tol
                max_err = max(max_err, abs(pq_feats[pid][k] - pt_feats[pid][k]))
        print(f"[1] iter round-trip: {n_iter}/{len(payloads)} rollouts  OK")
        print(f"[2] load_feature_cache parquet fallback: OK (tensor match)")
        print(f"[3] parquet vs .pt _rollout_feats: max_abs_err = {max_err:.2e} "
              f"(float16 .pt rounding) over {len(pq_feats)} pids")
        assert max_err < 5e-3, f"parquet vs pt feature mismatch too large ({max_err})"
        print("ALL CHECKS PASSED")


def test_cache_logits_parquet():
    _run()


if __name__ == "__main__":
    _run()
