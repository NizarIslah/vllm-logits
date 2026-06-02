"""Showcase 1 — three recoverability regimes, one screen.

Runs the full vllm_logits pipeline (cache_logits -> repair) on a small toy
problem set with a real model pair (Qwen3-0.6B specialist + Qwen3-0.6B-Base
ancestor), and prints, per failing problem, the junction-feature profile that
EXPLAINS the outcome plus the operator ladder (retry / rand / geo / local_temp).

Each row mirrors the paper's qualitative case studies:
    PROBLEM   outcome   rescued by   V_traj  V_t*  J_frac+  Jmax/mean  kl  G_cov  Δ_path   ladder

Run (1 GPU):
    export VLLM_WORKER_MULTIPROC_METHOD=spawn NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1
    PYTHONPATH=src python examples/showcase_three_regimes.py
"""
import os
import tempfile

import numpy as np

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")

from vllm_logits import LogitPipeline, numeric_answer

SPEC = os.environ.get(
    "VLLM_LOGITS_SPEC",
    "Qwen/Qwen3-0.6B",
)
ANC = os.environ.get(
    "VLLM_LOGITS_ANC",
    "Qwen/Qwen3-0.6B-Base",
)


def _toy_problems():
    """Arithmetic spanning easy->very-hard so all three regimes appear: easy items
    are sampling-variance failures (retry rescues = SAMPLING), medium are steerable
    (STEERABLE), and the large-multiplication / nested items are unrecoverable for a 0.6B
    where the base ancestor is also wrong, so steering has no correct target (HARD)."""
    items = [
        # easy (expect SAMPLING — retry rescues)
        ("p1", "Compute 48 + 79. Put the final answer in \\boxed{}.", "127"),
        ("p2", "Compute 256 - 89. Put the final answer in \\boxed{}.", "167"),
        ("p3", "Compute 144 / 12. Put the final answer in \\boxed{}.", "12"),
        ("p4", "Compute 13 * 13. Put the final answer in \\boxed{}.", "169"),
        # medium (expect STEERABLE — steerable)
        ("p5", "Compute 17 * 23. Put the final answer in \\boxed{}.", "391"),
        ("p6", "Compute 1000 - 333. Put the final answer in \\boxed{}.", "667"),
        ("p7", "Compute 84 * 36. Put the final answer in \\boxed{}.", "3024"),
        ("p8", "Compute 27 * 19 + 44. Put the final answer in \\boxed{}.", "557"),
        # hard (expect HARD — 0.6B and base both fail; no steerable correct target)
        ("p9", "Compute 739 * 856. Put the final answer in \\boxed{}.", "632584"),
        ("p10", "Compute 4127 * 638. Put the final answer in \\boxed{}.", "2633026"),
        ("p11", "Compute (473 + 588) * 67 - 1999. Put the final answer in \\boxed{}.", "69088"),
        ("p12", "Compute 98765 mod 137. Put the final answer in \\boxed{}.", "8"),
        ("p13", "Compute 6 ^ 7. Put the final answer in \\boxed{}.", "279936"),
        ("p14", "Compute 12345 * 6789. Put the final answer in \\boxed{}.", "83810205"),
    ]
    return [{"problem_id": pid, "prompt": q, "answer": a} for pid, q, a in items]


def _make_failing_rollouts(problems, pipe, k=2):
    """Generate k specialist rollouts per problem; keep only the failures."""
    from vllm import LLM, SamplingParams
    llm = LLM(model=pipe.specialist, dtype="bfloat16", enforce_eager=True,
              gpu_memory_utilization=0.45, max_model_len=1024,
              tensor_parallel_size=1, trust_remote_code=True)
    sp = SamplingParams(temperature=0.8, max_tokens=256, n=k)
    outs = llm.generate([p["prompt"] for p in problems], sp)
    chk = numeric_answer("answer")
    rollouts = []
    for p, o in zip(problems, outs):
        for ridx, comp in enumerate(o.outputs):
            rollouts.append({
                "problem_id": p["problem_id"], "rollout_idx": ridx,
                "generated_text": comp.text,
                "is_correct": chk(p, comp.text),
            })
    del llm
    import torch; torch.cuda.empty_cache()
    return rollouts


def _profile(pipe, pid):
    """Trajectory + junction feature profile for one pid from the cache."""
    from vllm_logits.features import _load_cell_rollouts, _rollout_feats
    cache_dir = str(pipe.store.feature_cache_dir(pipe.task, pipe.model_tag))
    rows = _load_cell_rollouts(cache_dir).get(str(pid), [])
    feats = [_rollout_feats(d, w=8) for d in rows]
    feats = [f for f in feats if f]
    if not feats:
        return None
    agg = {k: float(np.mean([f[k] for f in feats])) for k in feats[0]}
    return agg


def main():
    problems = _toy_problems()
    pipe = LogitPipeline(
        specialist=SPEC, ancestor=ANC, arch="auto",
        T=0.6, alpha=0.7, continuation_mode="temperature",
        cache_dir=tempfile.mkdtemp(prefix="showcase_"),
        cache_format="parquet", task="toy_arith", model_tag="qwen3_0p6b",
        max_new_tokens=256, max_model_len=1024, gpu_memory_utilization=0.45,
    )
    chk = numeric_answer("answer")

    print("[showcase] generating failing rollouts (specialist, T=0.8) ...")
    rollouts = _make_failing_rollouts(problems, pipe, k=8)
    n_fail = sum(1 for r in rollouts if not r["is_correct"])
    print(f"[showcase] {n_fail}/{len(rollouts)} rollouts failed; running pipeline ...")

    print("[showcase] stage 1: cache_logits ...")
    pipe.cache_logits(problems, rollouts, chk)
    print("[showcase] stage 2: repair (operator sweep) ...")
    results = pipe.repair(problems, rollouts, chk,
                          operators=["geo", "rand", "dense", "local_temp"],
                          k_values=[1, 5, 10])

    print("\n" + "=" * 100)
    print(f"{'PROBLEM':10} {'retry':6} {'rand':6} {'geoP':6} {'geoW':6} "
          f"{'dense':6} {'Ltemp':6}  {'V_traj':>7} {'V_junc':>7} {'kl_jc':>7} "
          f"{'Gcov_jc':>8} {'Dpath_jc':>9}")
    print("-" * 100)
    # Aggregate per pid across k_values: a condition counts as a rescue if it
    # succeeded at ANY k (so one clean row per problem, not one per k).
    by_pid = {}
    for res in results:
        agg = by_pid.setdefault(res["pid"], {})
        for c, d in res.items():
            if isinstance(d, dict) and "correct" in d:
                agg[c] = agg.get(c, False) or bool(d.get("correct"))
    regimes = {"SAMPLING": 0, "STEERABLE": 0, "HARD": 0}
    for pid in sorted(by_pid):
        agg = by_pid[pid]
        def ok(c):
            return "OK" if agg.get(c) else "-"
        ltemp_ok = any(v for k, v in agg.items() if k.startswith("local_temp_"))
        retry_ok = agg.get("retry", False)
        any_steer = any(agg.get(c) for c in ("rand", "geo_pred", "geo_wrong", "dense")) or ltemp_ok
        regime = "SAMPLING" if retry_ok else ("STEERABLE" if any_steer else "HARD")
        regimes[regime] += 1
        prof = _profile(pipe, pid) or {}
        print(f"{pid:10} {ok('retry'):6} {ok('rand'):6} {ok('geo_pred'):6} "
              f"{ok('geo_wrong'):6} {ok('dense'):6} {('OK' if ltemp_ok else '-'):6}  "
              f"{prof.get('logit_var_trace', 0):7.3f} {prof.get('logit_var_junc', 0):7.3f} "
              f"{prof.get('kl_div_junc', 0):7.2f} {prof.get('G_cov_junc', 0):8.3f} "
              f"{prof.get('Delta_path_junc', 0):9.3f}   [{regime}]")
    print("=" * 100)
    print(f"regime counts: {regimes}")
    print("[showcase] DONE")
    return results, regimes


if __name__ == "__main__":
    main()
