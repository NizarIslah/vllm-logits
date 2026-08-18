"""The worked example, on shipped data. No GPU, no model download, no torch.

    python -m vllm_logits.demo                 # all cells
    python -m vllm_logits.demo --cell sft0p6b|gsm8k
    python -m vllm_logits.demo --plot panel.png

It answers two questions on 1,423 real failed problems from four post-trained models across three
tasks (see `data/README.md` for provenance):

  1. Of the failures, how many are worth more sampling, how many need a different kind of
     intervention, and how many are beyond any of them?
  2. Does choosing the intervention per-failure from trace features beat committing to one
     intervention everywhere?
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from .routing import RoutingPolicy, route, route_scores

DATA = Path(__file__).resolve().parent / "data" / "failure_profiles.csv"

#: Interventions assayed on every problem, plus plain resampling as the baseline.
INTERVENTIONS = ["sparse_steer", "sparse_steer_random", "local_temp", "dense_steer"]
#: Pretty names, kept jargon-free for readers who have not read the paper.
PRETTY = {
    "sparse_steer": "sparse steer @ junction",
    "sparse_steer_random": "sparse steer @ random",
    "local_temp": "local temperature lift",
    "dense_steer": "dense steer (whole trace)",
}
#: Cell used for the routing half of the default run (see data/README.md for why).
FLAGSHIP = "sft0p6b|gsm8k"
#: Eq. 2 thresholds: an intervention must beat resampling by this much, on a problem this hard.
TAU_STEERABLE, TAU_HARD = 0.05, 0.50


def load(path: Path = DATA) -> dict[str, np.ndarray]:
    """Read the shipped per-problem profiles into column arrays (stdlib csv; no pandas)."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"no rows in {path}")
    num = set(rows[0]) - {"cell", "model", "task", "problem_id"}
    out: dict[str, np.ndarray] = {}
    for key in rows[0]:
        vals = [r[key] for r in rows]
        out[key] = np.array([float(v) for v in vals]) if key in num else np.array(vals)
    return out


def classify(d: dict[str, np.ndarray]) -> np.ndarray:
    """Label every failure: 'resample', 'steerable' or 'beyond reach'.

    steerable     resampling is inadequate AND some intervention beats it by >= 5 pp.
                  This is the population where choosing an intervention can pay off
    beyond reach  no intervention rescues it at all
    resample      neither: plain resampling already gets there
    """
    best = np.max([d[o] for o in INTERVENTIONS], axis=0)
    steerable = ((best - d["retry"]) >= TAU_STEERABLE) & ((1.0 - d["retry"]) >= TAU_HARD)
    beyond = (~steerable) & (best == 0.0)
    labels = np.full(len(best), "resample", dtype=object)
    labels[steerable] = "steerable"
    labels[beyond] = "beyond reach"
    return labels


def _bar(frac: float, width: int = 28) -> str:
    filled = int(round(frac * width))
    return "█" * filled + "·" * (width - filled)


def _select(d: dict[str, np.ndarray], cell: str | None) -> tuple[dict[str, np.ndarray], int]:
    sel = np.ones(len(d["retry"]), dtype=bool) if cell is None else (d["cell"] == cell)
    if not sel.any():
        raise SystemExit(f"no rows for cell {cell!r}; available: {sorted(set(d['cell']))}")
    return {k: v[sel] for k, v in d.items()}, int(sel.sum())


def taxonomy(d: dict[str, np.ndarray], cell: str | None = None) -> dict:
    """Part 1: of the failures, which are worth more compute at all?"""
    sub, n = _select(d, cell)
    labels = classify(sub)
    scope = "all cells" if cell is None else cell
    print(f"\n{'='*78}\n  Failed generations: what is worth more compute?   [{scope}, n={n}]\n{'='*78}\n")
    counts = Counter(labels)
    for name, blurb in (
        ("resample", "plain resampling already gets there, so spend samples"),
        ("steerable", "resampling stalls, but an intervention rescues it: spend differently"),
        ("beyond reach", "nothing we tried rescues it, so stop spending"),
    ):
        c = counts.get(name, 0)
        print(f"  {name:<13} {_bar(c/n)}  {100*c/n:4.1f}%  ({c:4d})   {blurb}")
    return {"n": n, "counts": dict(counts)}


def routing_report(d: dict[str, np.ndarray], cell: str | None = None) -> dict:
    """Part 2: for the steerable ones, does per-failure routing beat one fixed choice?"""
    sub, _ = _select(d, cell)
    labels = classify(sub)
    steer = labels == "steerable"
    st = {k: v[steer] for k, v in sub.items()}
    scope = "all cells" if cell is None else cell
    print(f"\n{'-'*78}\n  Which intervention?   [{scope}, {int(steer.sum())} steerable failures]\n{'-'*78}")
    ops = route(st)
    z = route_scores(st)
    print("\n  Routed from trace features alone. No repair outcomes used.\n")
    print(f"  {'routed to':<28}{'n':>5}   {'spread':>8}{'concentr.':>11}{'dispersion':>12}")
    for op in RoutingPolicy().operators + [RoutingPolicy().fallback]:
        m = ops == op
        if not m.any():
            continue
        print(f"  {PRETTY.get(op, op):<28}{int(m.sum()):>5}   "
              f"{z['spread'][m].mean():>8.2f}{z['concentration'][m].mean():>11.2f}"
              f"{z['logit_dispersion'][m].mean():>12.2f}")
    print("\n  Each group's own signature feature is highest. That is the rule working.")

    # --- routed vs. committing to one intervention everywhere ---
    routed = np.array([st[o][i] for i, o in enumerate(ops)])
    fixed = {o: st[o].mean() for o in INTERVENTIONS}
    best_fixed = max(fixed, key=fixed.get)
    print("\n  Rescue rate, best-of-3 attempts:\n")
    print(f"    {'routed per failure':<34}{routed.mean():6.1%}   <-- the feature rule")
    for o in sorted(fixed, key=fixed.get, reverse=True):
        flag = "   <-- best single choice" if o == best_fixed else ""
        print(f"    {'always ' + PRETTY.get(o, o):<34}{fixed[o]:6.1%}{flag}")
    print(f"    {'resample again':<34}{st['retry'].mean():6.1%}")
    delta = 100 * (routed.mean() - fixed[best_fixed])
    verdict = ("routing beats every fixed choice" if delta >= 0
               else f"the fixed choice '{PRETTY.get(best_fixed, best_fixed)}' wins here")
    print(f"\n  -> {verdict} ({delta:+.1f} pp vs. the best single choice).")
    return {"scope": scope, "n_steerable": int(steer.sum()),
            "routed": float(routed.mean()),
            "fixed": fixed, "best_fixed": best_fixed, "delta_pp": float(delta),
            "ops": ops, "z": z, "labels": labels[steer]}


def plot(res: dict, path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"\n  [skipped {path}] plotting needs matplotlib: pip install \"vllm-logits[demo]\"")
        return False
    z, ops = res["z"], res["ops"]
    x = z["concentration"] - z["spread"]      # sparse-vs-dense axis
    y = z["logit_dispersion"]                 # temperature-sensitivity axis
    fig, ax = plt.subplots(figsize=(7.4, 5.6), dpi=150)
    colors = {"dense_steer": "#4C6EF5", "sparse_steer": "#F76707",
              "local_temp": "#12B886", "sparse_steer_random": "#868E96"}
    for op in dict.fromkeys(ops):
        m = ops == op
        ax.scatter(x[m], y[m], s=52, alpha=.85, edgecolor="white", linewidth=.7,
                   c=colors.get(op, "#868E96"), label=f"{PRETTY.get(op, op)}  (n={int(m.sum())})")
    ax.axhline(0, color="#CED4DA", lw=.8, zorder=0)
    ax.axvline(0, color="#CED4DA", lw=.8, zorder=0)
    ax.set_xlabel("← divergence spread broadly          divergence concentrated at one point →",
                  fontsize=10)
    ax.set_ylabel("responds to temperature  →", fontsize=10)
    scope = res.get("scope", "")
    n = res.get("n_steerable")
    where = f" ({scope})" if scope and scope != "all cells" else ""
    ax.set_title(f"Which of {n} failures gets which intervention{where},\n"
                 "decided from the failed generation alone",
                 fontsize=12.5, pad=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0, -0.16), ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"\n  [wrote] {path}")
    return True


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cell", default=None,
                    help="restrict to one model|task cell, e.g. 'sft0p6b|gsm8k'")
    ap.add_argument("--data", type=Path, default=DATA, help="override the shipped CSV")
    ap.add_argument("--plot", type=Path, nargs="?", const=Path("routing_panel.png"),
                    default=None, help="also write the routing panel (needs matplotlib)")
    ap.add_argument("--list-cells", action="store_true")
    args = ap.parse_args(argv)

    d = load(args.data)
    if args.list_cells:
        for c in sorted(set(d["cell"])):
            print(f"{c}  n={int((d['cell']==c).sum())}")
        return
    if args.cell:
        taxonomy(d, args.cell)
        res = routing_report(d, args.cell)
    else:
        # Default: the taxonomy is a population-level fact, so it is shown over all
        # 1,423 failures. Routing is shown on one cell, because which intervention
        # wins is model- and task-dependent -- see the scope note below.
        taxonomy(d, None)
        res = routing_report(d, FLAGSHIP)
        print(f"\n  Scope: routing is shown on {FLAGSHIP}. Which intervention wins is model- and")
        print( "  task-dependent, so run --cell <cell> on your own cells rather than assuming this")
        print( "  ordering transfers. --list-cells shows what ships here.")
    if args.plot:
        plot(res, args.plot)
    print()


if __name__ == "__main__":
    main()
