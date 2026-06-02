"""vllm-logits — standalone dual-model logit-intervention toolkit for vLLM.

Public API:
    LogitPipeline(specialist, ancestor, ...).run(problems, rollouts, correctness_fn,
                                                  operators, k_values)
        cache_logits → repair, for any (model, task). Also .cache_logits / .repair.

Tier-0 primitives (storage, alpha, junctions, io correctness defaults) import
ONLY torch / numpy / polars and are usable without vLLM. The heavier symbols
(LogitPipeline, engines, processors, backbones) import vLLM lazily, so a CPU-only
environment can still `import vllm_logits` and use the Tier-0 pieces.
"""

__version__ = "0.1.0"

# Tier 0 — no vLLM required
from .storage import LogitStore
from .alpha import entropy_gap, chi2_divergence, adaptive_alpha, mixing_diagnostics
from .junctions import (
    JunctionContext, RepairContext,
    DemotionScoreDetector, ProbCoverageDetector, RandomPositionDetector,
    MatchedPositionDetector, PageHinkleyDetector, calibrate_ph_threshold,
    find_firing_position,
)
from .io import (
    Problem, Rollout, load_problems, load_rollouts, load_jsonl, save_jsonl,
    exact_match, numeric_answer, regex,
)

__all__ = [
    "__version__",
    "LogitPipeline",
    # storage
    "LogitStore",
    # alpha
    "entropy_gap", "chi2_divergence", "adaptive_alpha", "mixing_diagnostics",
    # junctions
    "JunctionContext", "RepairContext",
    "DemotionScoreDetector", "ProbCoverageDetector", "RandomPositionDetector",
    "MatchedPositionDetector", "PageHinkleyDetector", "calibrate_ph_threshold",
    "find_firing_position",
    # io
    "Problem", "Rollout", "load_problems", "load_rollouts", "load_jsonl",
    "save_jsonl", "exact_match", "numeric_answer", "regex",
    # scoring math
    "compute_scores_at_t", "estimate_background",
]


def __getattr__(name):
    """Lazy import of vLLM-dependent symbols (keeps Tier-0 import vLLM-free)."""
    if name == "LogitPipeline":
        from .pipeline import LogitPipeline
        return LogitPipeline
    if name in ("compute_scores_at_t", "estimate_background"):
        from . import scoring
        return getattr(scoring, name)
    if name == "VllmCacheLogitsEngine":
        from .features import VllmCacheLogitsEngine
        return VllmCacheLogitsEngine
    if name == "LogitRepairEngine":
        from .repair import LogitRepairEngine
        return LogitRepairEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
