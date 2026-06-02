"""Showcase 2 — population view: feature-routed operator vs. empirical recoverability.

Takes a failure set spanning easy->hard problems, runs the full pipeline
(cache_logits -> repair), and renders ONE panel over a 2-D projection of three
problem-level trajectory features, communicating both signals on every point:

  color  the operator the prospective routing rule picks from the features alone
         (argmax of three z-scored features, no outcomes):
             dense steer | sparse steer | local temperature lift
  label  the empirical recoverability of that failure, from the repair outcomes:
             retry-solvable  (plain resampling rescues it) or
             hard            (it does not, so a logit intervention or a
                              local-temperature lift is what can still help --
                              the paper's routing target)

The three routing features and the operator each one makes actionable:
    spread        = J_frac+  (how broad the divergence is)      -> dense steer (DL)
    concentration = log10(J_max / J_mean)  (one sharp spike)     -> sparse steer (SL-G)
    logit dispersion = log10(V_t*)  (variance of logits at the spike,  -> local temperature lift
                       a.k.a. temperature sensitivity)

The story: a rule reading only the features (color) prescribes an operator, and the
empirical label (retry-solvable vs. hard) shows which failures actually need one.
Results are cached to disk (VLLM_LOGITS_CLUSTER_DATA, default
docs/clustering_data.json), so re-plotting needs no GPU; set VLLM_LOGITS_FORCE=1 to
recompute. Saves docs/clustering.png.

Run (1 GPU, first time):
    export VLLM_WORKER_MULTIPROC_METHOD=spawn NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1
    PYTHONPATH=src python examples/showcase_clustering.py
"""
import json
import os
import tempfile

import numpy as np

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")

from vllm_logits import LogitPipeline, numeric_answer
from vllm_logits.features import _load_cell_rollouts

SPEC = os.environ.get("VLLM_LOGITS_SPEC", "Qwen/Qwen3-0.6B")
ANC = os.environ.get("VLLM_LOGITS_ANC", "Qwen/Qwen3-0.6B-Base")

# The three paper-derived problem-level features (Box 1 / prospective rule), each
# mapping to the one operator it makes actionable. The rule routes every failed pid
# to the operator whose z-scored feature is largest (argmax), with no gate:
#   spread           = J_frac+  (fraction of trace tokens with J_approx > 0)     -> dense steer (DL)
#   concentration    = log10(J_max / J_mean)  (one sharp spike vs. diffuse)      -> sparse steer (SL-G)
#   logit dispersion = log10(V_t*)  (variance of the specialist's logits at the  -> local temperature lift (T_loc)
#                      spike; a.k.a. temperature sensitivity / Fisher info of the temperature submodel)
_FEATS = ["spread", "concentration", "logit dispersion"]
_FEAT_OP = ["dense steer", "sparse steer", "local temperature lift"]  # by feature index

# Per-point empirical recoverability (best-of-3), shown as the point label:
#   retry-solvable = plain resampling rescues it (no routing needed)
#   hard           = it does not, so a logit intervention or local-temperature
#                    lift is what can still help (the paper's routing target).
_RECO_ORDER = ["retry-solvable", "hard"]

# Right panel: feature-routed operator.
_OP_ORDER = ["sparse steer", "dense steer", "local temperature lift"]
_OP_COLOR = {"sparse steer": "#3b6ea5", "dense steer": "#6a4c93",
             "local temperature lift": "#3aa6a0"}


def _spanning_problems():
    """21 arithmetic problems, 7 easy / 7 medium / 7 hard, so all three regimes
    appear. The larger set gives tighter, better-separated clusters."""
    items = [
        # easy
        ("p1", "Compute 48 + 79. Put the final answer in \\boxed{}.", "127"),
        ("p2", "Compute 256 - 89. Put the final answer in \\boxed{}.", "167"),
        ("p3", "Compute 144 / 12. Put the final answer in \\boxed{}.", "12"),
        ("p4", "Compute 13 * 13. Put the final answer in \\boxed{}.", "169"),
        ("p5", "Compute 91 + 18. Put the final answer in \\boxed{}.", "109"),
        ("p6", "Compute 300 - 176. Put the final answer in \\boxed{}.", "124"),
        ("p7", "Compute 15 * 6. Put the final answer in \\boxed{}.", "90"),
        # medium
        ("p8", "Compute 17 * 23. Put the final answer in \\boxed{}.", "391"),
        ("p9", "Compute 1000 - 333. Put the final answer in \\boxed{}.", "667"),
        ("p10", "Compute 84 * 36. Put the final answer in \\boxed{}.", "3024"),
        ("p11", "Compute 27 * 19 + 44. Put the final answer in \\boxed{}.", "557"),
        ("p12", "Compute 56 * 47. Put the final answer in \\boxed{}.", "2632"),
        ("p13", "Compute 123 + 456 + 789. Put the final answer in \\boxed{}.", "1368"),
        ("p14", "Compute 72 * 38. Put the final answer in \\boxed{}.", "2736"),
        # hard
        ("p15", "Compute 739 * 856. Put the final answer in \\boxed{}.", "632584"),
        ("p16", "Compute 4127 * 638. Put the final answer in \\boxed{}.", "2633026"),
        ("p17", "Compute (473 + 588) * 67 - 1999. Put the final answer in \\boxed{}.", "69088"),
        ("p18", "Compute 6 ^ 7. Put the final answer in \\boxed{}.", "279936"),
        ("p19", "Compute 12345 * 6789. Put the final answer in \\boxed{}.", "83810205"),
        ("p20", "Compute 857 * 643. Put the final answer in \\boxed{}.", "551051"),
        ("p21", "Compute 9999 * 8888. Put the final answer in \\boxed{}.", "88871112"),
    ]
    return [{"problem_id": pid, "prompt": q, "answer": a} for pid, q, a in items]


def _make_failing_rollouts(pipe, problems, k=8):
    from vllm import LLM, SamplingParams
    import torch
    llm = LLM(model=pipe.specialist, dtype="bfloat16", enforce_eager=True,
              gpu_memory_utilization=0.45, max_model_len=1024,
              tensor_parallel_size=1, trust_remote_code=True)
    # T=0.6 to match the retry baseline: a failure and its retry are draws from
    # the same distribution, so the "retry" outcome is a clean comparison.
    sp = SamplingParams(temperature=0.6, max_tokens=256, n=k)
    outs = llm.generate([p["prompt"] for p in problems], sp)
    chk = numeric_answer("answer")
    rollouts = []
    for p, o in zip(problems, outs):
        for ridx, comp in enumerate(o.outputs):
            rollouts.append({"problem_id": p["problem_id"], "rollout_idx": ridx,
                             "generated_text": comp.text, "is_correct": chk(p, comp.text)})
    del llm
    torch.cuda.empty_cache()
    return rollouts


def _classes_rescued(agg):
    """Which operator classes rescue this pid (best-of-3). retry is included."""
    res = []
    if agg.get("retry"):
        res.append("retry")
    if any(agg.get(c) for c in ("geo_pred", "geo_wrong", "rand")):
        res.append("sparse steer")
    if agg.get("dense"):
        res.append("dense steer")
    if any(v for k, v in agg.items() if k.startswith("local_temp_")):
        res.append("local temperature lift")
    return res


def _paper_features(rows, w=8):
    """Per-pid (spread, log concentration, log logit-dispersion), rollout-averaged.

    spread        = mean_t 1[J_approx_t > 0]            (deformation spread, J_frac+)
    concentration = log10( max_t J / mean_t J )         (peak-to-mean junction ratio)
    logit dispersion = log10( mean logit_var in +/-w window around argmax J )  (V_t*)
    Trim trailing padding by kl_div==0, mirroring the real caches.
    """
    sp, lc, lv = [], [], []
    for d in rows:
        kl = d["kl_div"].float().numpy()
        valid = kl != 0.0
        hi = int(np.where(valid)[0][-1]) + 1 if valid.any() else len(kl)
        J = d["J_approx"].float().numpy()[:hi]
        V = d["logit_var"].float().numpy()[:hi]
        if len(J) == 0:
            continue
        j = int(np.argmax(J)); lo, up = max(0, j - w), min(len(J), j + w + 1)
        jmean = max(float(J.mean()), 1e-8)
        sp.append(float((J > 0).mean()))
        lc.append(float(np.log10(max(float(J.max()) / jmean, 1e-8))))
        lv.append(float(np.log10(max(float(V[lo:up].mean()), 1e-8))))
    if not sp:
        return None
    return [float(np.mean(sp)), float(np.mean(lc)), float(np.mean(lv))]


def _compute(data_path):
    """Run the GPU pipeline (cache_logits -> repair), build per-pid records, and
    cache them to data_path so future runs can re-plot without a GPU."""
    problems = _spanning_problems()
    pipe = LogitPipeline(
        specialist=SPEC, ancestor=ANC, arch="auto", T=0.6, alpha=0.7,
        continuation_mode="temperature",
        cache_dir=tempfile.mkdtemp(prefix="cluster_"),
        cache_format="parquet", task="toy_arith", model_tag="qwen3_0p6b",
        max_new_tokens=256, max_model_len=1024, gpu_memory_utilization=0.45,
    )
    chk = numeric_answer("answer")

    print("[showcase] generating failing rollouts (specialist, T=0.6) ...")
    rollouts = _make_failing_rollouts(pipe, problems, k=8)
    print("[showcase] stage 1: cache_logits ...")
    pipe.cache_logits(problems, rollouts, chk)
    print("[showcase] stage 2: repair (operator sweep) ...")
    # Rescue budget = best-of-3 (3 tries per operator; retry = 3 resamples).
    # A fixed small budget, in the spirit of the paper's repair@3, so the
    # separability split is informative rather than washed out by best-of-10.
    results = pipe.repair(problems, rollouts, chk,
                          operators=["geo", "rand", "dense", "local_temp"],
                          k_values=[1, 3])

    by_pid = {}
    for res in results:
        agg = by_pid.setdefault(res["pid"], {})
        for c, d in res.items():
            if isinstance(d, dict) and "correct" in d:
                agg[c] = agg.get(c, False) or bool(d.get("correct"))

    cache_dir = str(pipe.store.feature_cache_dir(pipe.task, pipe.model_tag))
    cell = _load_cell_rollouts(cache_dir)
    records = []
    for pid in sorted(set(map(str, by_pid)) & set(map(str, cell))):
        feats = _paper_features(cell[pid], w=8)
        if feats is None:
            continue
        records.append({"pid": pid, "feats": feats,
                        "rescuers": _classes_rescued(by_pid[pid])})

    os.makedirs(os.path.dirname(data_path) or ".", exist_ok=True)
    with open(data_path, "w") as fh:
        json.dump(records, fh, indent=2)
    print(f"[showcase] saved {len(records)} records -> {data_path}")
    return records


def _plot(records, out_png):
    if len(records) < 4:
        print(f"[showcase] only {len(records)} usable pids; need >=4.")
        return
    X = np.array([r["feats"] for r in records], dtype=float)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-8)  # z-normalize each feature

    # Right panel: prospective rule (argmax of z-scored features), no gate.
    routed = Z.argmax(1)
    ops = [_FEAT_OP[i] for i in routed]

    # Per-point recoverability label (empirical, best-of-3): retry-solvable if retry
    # rescues it, else hard (retry insufficient -> needs a logit intervention or a
    # local-temperature lift). Shown as the point label on the single panel.
    reco = ["retry-solvable" if "retry" in r["rescuers"] else "hard" for r in records]

    # ---- text summary ----
    print("\n" + "=" * 92)
    print("routing features (z-scored):  " + " | ".join(
        f"{_FEATS[i]} -> {_FEAT_OP[i]}" for i in range(len(_FEATS))))
    print("-" * 92)
    print(f"{'routed operator':24} {'n':>3}  " + "  ".join(f"{nm:>16}" for nm in _FEATS))
    for op in _OP_ORDER:
        idx = _FEAT_OP.index(op)
        m = routed == idx
        n = int(m.sum())
        cen = Z[m].mean(0) if n else np.full(len(_FEATS), float("nan"))
        print(f"{op:24} {n:>3}  " + "  ".join(f"{v:16.2f}" for v in cen))
    print("-" * 92)
    rc = {s: reco.count(s) for s in _RECO_ORDER}
    print(f"recoverability (empirical): {rc}")
    print(f"over {len(records)} failing problems")
    print("=" * 92)

    # ---- two-subplot figure over the same projection ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 2-D projection: PCA of the z-normalized 3-feature space.
        _, _, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
        P = (Z - Z.mean(0)) @ Vt[:2].T

        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        fig, ax = plt.subplots(figsize=(9.0, 7.0), constrained_layout=True)

        # Color = feature-routed operator; text label = empirical recoverability.
        for cat in _OP_ORDER:
            m = np.array([o == cat for o in ops])
            if not m.any():
                continue
            ax.scatter(P[m, 0], P[m, 1], s=200, color=_OP_COLOR[cat],
                       edgecolor="#333", linewidth=0.8, alpha=0.92,
                       label=f"{cat}  (n={int(m.sum())})")
        for i, lbl in enumerate(reco):
            ax.annotate(lbl, (P[i, 0], P[i, 1]), fontsize=6.5, ha="center",
                        va="bottom", xytext=(0, 9), textcoords="offset points",
                        color="#222")
        ax.set_title("Each failure: routed operator (color) + retry-solvable vs. hard (label)\n"
                     f"prospective rule routes by argmax of 3 features  ·  {len(records)} failures",
                     fontsize=12)
        ax.set_xlabel("PC1  (projection of spread / concentration / logit dispersion)", fontsize=9.5)
        ax.set_ylabel("PC2", fontsize=9.5)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(title="feature-routed operator", fontsize=9, title_fontsize=9,
                  loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=False)
        fig.savefig(out_png, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"[showcase] saved plot -> {out_png}")
    except Exception as e:
        print(f"[showcase] plot skipped ({e})")
    print("[showcase] DONE")


def main():
    out_png = os.environ.get("VLLM_LOGITS_CLUSTER_PNG", "docs/clustering.png")
    data_path = os.environ.get("VLLM_LOGITS_CLUSTER_DATA", "docs/clustering_data.json")
    force = os.environ.get("VLLM_LOGITS_FORCE", "0") == "1"
    if os.path.exists(data_path) and not force:
        with open(data_path) as fh:
            records = json.load(fh)
        print(f"[showcase] loaded {len(records)} records from {data_path} "
              f"(GPU pipeline skipped; set VLLM_LOGITS_FORCE=1 to recompute)")
    else:
        records = _compute(data_path)
    _plot(records, out_png)
    return records


if __name__ == "__main__":
    main()
