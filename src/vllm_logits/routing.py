"""Prospective routing: pick a repair operator from a failed trace's features alone.

Three trajectory features, each measuring one thing an operator class can act on:

    spread            how broadly the fine-tune diverged from its base   -> dense steer
    concentration     one sharp divergence spike vs. diffuse             -> sparse steer
    logit_dispersion  how strongly the token responds to temperature     -> local temperature lift

The rule z-scores the three features across the population and routes each failure to the operator
whose feature is largest. It is prospective: it reads only the failed generation, never the outcome of
any repair attempt, so it is usable at deployment time on failures you have not tried to fix yet.

This module is Tier 0: numpy only, no vLLM, no GPU. You can route failures on a laptop from features
you extracted elsewhere.

    from vllm_logits import route
    ops = route({"spread": [...], "concentration": [...], "logit_dispersion": [...]})

Supply your own feature->operator mapping (or your own feature names) with `RoutingPolicy`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: The default rule: each feature routes to the operator its geometry makes actionable.
DEFAULT_MAPPING: dict[str, str] = {
    "spread": "dense_steer",
    "concentration": "sparse_steer",
    "logit_dispersion": "local_temp",
}

#: Used when no feature stands out (every z-score <= 0): a position-agnostic hedge.
DEFAULT_FALLBACK = "sparse_steer_random"


@dataclass(frozen=True)
class RoutingPolicy:
    """A feature -> operator rule.

    mapping   feature name -> operator name. Order is irrelevant; ties break by first key.
    fallback  operator for items whose largest z-score is <= `fallback_below` (no feature stands
              out, so the geometry prescribes nothing). Set to None to disable and always route to
              the argmax.
    fallback_below
              the z threshold under which `fallback` fires. 0.0 means "no feature is above the
              population average".
    """

    mapping: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MAPPING))
    fallback: str | None = DEFAULT_FALLBACK
    fallback_below: float = 0.0

    @property
    def features(self) -> list[str]:
        return list(self.mapping)

    @property
    def operators(self) -> list[str]:
        return list(self.mapping.values())


def zscore(x: np.ndarray) -> np.ndarray:
    """Population z-score, safe on a constant column (returns zeros rather than NaN)."""
    x = np.asarray(x, dtype=float)
    sd = x.std()
    if not np.isfinite(sd) or sd == 0.0:
        return np.zeros_like(x)
    return (x - x.mean()) / sd


def route(features, policy: RoutingPolicy | None = None) -> np.ndarray:
    """Route each failure to an operator.

    features  a mapping {feature_name: sequence of values}, a pandas DataFrame, or anything
              indexable by the policy's feature names. All features must be the same length.
    policy    a RoutingPolicy; the default rule is used if omitted.

    Returns an array of operator names, one per failure.

    The z-scores are computed over whatever population you pass in. Pass all your failures at once
    for a single consistent scaling; pass one cell at a time to scale within that cell.
    """
    policy = policy or RoutingPolicy()
    cols = [zscore(_column(features, f)) for f in policy.features]
    n = {len(c) for c in cols}
    if len(n) != 1:
        raise ValueError(f"features have mismatched lengths: {[len(c) for c in cols]}")
    Z = np.stack(cols, axis=1)                       # (n_items, n_features)
    ops = np.array([policy.mapping[f] for f in policy.features])
    chosen = ops[Z.argmax(axis=1)]
    if policy.fallback is not None:
        chosen = np.where(Z.max(axis=1) <= policy.fallback_below, policy.fallback, chosen)
    return chosen


def route_scores(features, policy: RoutingPolicy | None = None) -> dict[str, np.ndarray]:
    """The z-scored features behind a routing decision, keyed by feature name.

    Useful for inspecting *why* a failure routed the way it did, and for plotting.
    """
    policy = policy or RoutingPolicy()
    return {f: zscore(_column(features, f)) for f in policy.features}


def _column(features, name: str) -> np.ndarray:
    try:
        return np.asarray(features[name], dtype=float)
    except (KeyError, IndexError, TypeError) as exc:
        raise KeyError(
            f"routing feature {name!r} not found in the supplied features. "
            f"Provide it, or pass a RoutingPolicy with your own feature names."
        ) from exc
