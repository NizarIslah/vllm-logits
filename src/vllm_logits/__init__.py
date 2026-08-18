"""vllm-logits: decide whether a failed generation is worth more compute, and what to spend it on.

Load a *specialist* (your fine-tune) and its *ancestor* (the base/reference it came from) into a
single vLLM model, read per-token divergence off a failed trace in one pass, and either steer the
decode back toward the suppressed alternative or conclude that no amount of resampling will help.

Three dependency tiers, so the cheap parts stay cheap:

  Tier 0  numpy only. Always importable, no GPU, no model.
            route / RoutingPolicy / route_scores   pick an operator from features
            Problem / Rollout / load_* / save_*    the input contract
            exact_match / numeric_answer / regex   correctness helpers
            demo                                   the worked example on shipped data

  Tier 1  + torch  (`pip install "vllm-logits[engine]"`)
            LogitStore, junction detectors, alpha signals, scoring

  Tier 2  + vLLM   (`pip install "vllm-logits[engine]"`)
            LogitPipeline, engines, processors, dual backbones

Tier 1 and Tier 2 symbols are imported lazily, so a laptop with numpy can `import vllm_logits`,
route real failures, and run `python -m vllm_logits.demo` without installing torch or vLLM.
"""

__version__ = "0.2.0"

# ---------------------------------------------------------------------------
# Tier 0: numpy only
# ---------------------------------------------------------------------------
from .routing import (
    DEFAULT_MAPPING, DEFAULT_FALLBACK, RoutingPolicy, route, route_scores, zscore,
)
from .io import (
    Problem, Rollout, load_problems, load_rollouts, load_jsonl, save_jsonl,
    exact_match, numeric_answer, regex,
)

__all__ = [
    "__version__",
    # Tier 0: routing
    "route", "route_scores", "zscore", "RoutingPolicy",
    "DEFAULT_MAPPING", "DEFAULT_FALLBACK",
    # Tier 0: input contract
    "Problem", "Rollout", "load_problems", "load_rollouts", "load_jsonl",
    "save_jsonl", "exact_match", "numeric_answer", "regex",
    # Tier 1: torch
    "LogitStore",
    "entropy_gap", "chi2_divergence", "adaptive_alpha", "mixing_diagnostics",
    "JunctionContext", "RepairContext",
    "DemotionScoreDetector", "ProbCoverageDetector", "RandomPositionDetector",
    "MatchedPositionDetector", "PageHinkleyDetector", "calibrate_ph_threshold",
    "find_firing_position",
    "compute_scores_at_t", "estimate_background",
    # Tier 2: vLLM
    "LogitPipeline", "VllmCacheLogitsEngine", "LogitRepairEngine",
]

# ---------------------------------------------------------------------------
# Tier 1 and Tier 2: lazy, with an actionable message when the extra is missing
# ---------------------------------------------------------------------------
_TIER1 = {
    "LogitStore": "storage",
    "entropy_gap": "alpha", "chi2_divergence": "alpha",
    "adaptive_alpha": "alpha", "mixing_diagnostics": "alpha",
    "JunctionContext": "junctions", "RepairContext": "junctions",
    "DemotionScoreDetector": "junctions", "ProbCoverageDetector": "junctions",
    "RandomPositionDetector": "junctions", "MatchedPositionDetector": "junctions",
    "PageHinkleyDetector": "junctions", "calibrate_ph_threshold": "junctions",
    "find_firing_position": "junctions",
    "compute_scores_at_t": "scoring", "estimate_background": "scoring",
}
_TIER2 = {
    "LogitPipeline": "pipeline",
    "VllmCacheLogitsEngine": "features",
    "LogitRepairEngine": "repair",
}


def __getattr__(name):
    """Import Tier 1/2 symbols on demand; explain the fix if the extra is not installed."""
    module = _TIER1.get(name) or _TIER2.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    tier = "torch" if name in _TIER1 else "vLLM"
    try:
        from importlib import import_module
        return getattr(import_module(f".{module}", __name__), name)
    except ImportError as exc:
        raise ImportError(
            f"{name} needs {tier}, which is not installed.\n"
            f'    pip install "vllm-logits[engine]"\n'
            f"Tier-0 features (route, the input contract, python -m vllm_logits.demo) "
            f"work without it."
        ) from exc
