"""Tier-0 routing tests — numpy only, no torch, no vLLM, no GPU."""
import numpy as np
import pytest

from vllm_logits.routing import (
    DEFAULT_FALLBACK, DEFAULT_MAPPING, RoutingPolicy, route, route_scores, zscore,
)


def test_each_feature_routes_to_its_operator():
    # one item per feature, each dominant on exactly that feature
    feats = {"spread": [3, 0, 0], "concentration": [0, 3, 0], "logit_dispersion": [0, 0, 3]}
    assert list(route(feats)) == [
        DEFAULT_MAPPING["spread"],
        DEFAULT_MAPPING["concentration"],
        DEFAULT_MAPPING["logit_dispersion"],
    ]


def test_fallback_fires_when_no_feature_stands_out():
    # the last item is below the mean on every feature, so nothing is prescribed
    feats = {"spread": [3, 0, 0, -1], "concentration": [0, 3, 0, -1], "logit_dispersion": [0, 0, 3, -1]}
    assert route(feats)[-1] == DEFAULT_FALLBACK


def test_fallback_can_be_disabled():
    feats = {"spread": [1, -1], "concentration": [0, -2], "logit_dispersion": [0, -3]}
    out = route(feats, RoutingPolicy(fallback=None))
    assert DEFAULT_FALLBACK not in out


def test_custom_policy_features_and_operators():
    pol = RoutingPolicy(mapping={"a": "op_a", "b": "op_b"}, fallback=None)
    assert list(route({"a": [1, 0], "b": [0, 1]}, pol)) == ["op_a", "op_b"]


def test_scaling_is_relative_to_the_population_passed_in():
    """The same raw features can route differently depending on the population.

    Item 0 is identical in both calls. In A its spread and concentration tie, so the first
    mapped feature wins. In B another item stretches the spread column, which pushes item 0's
    spread z-score down and lets concentration overtake it. This is by design -- the rule is
    population-relative -- and it is why the docstring tells callers to pass all their failures
    at once, or one cell at a time, deliberately.
    """
    a = route({"spread": [1, 0, 0], "concentration": [1, 0, 0], "logit_dispersion": [0, 0, 0]})
    b = route({"spread": [1, 0, 20], "concentration": [1, 0, 0], "logit_dispersion": [0, 0, 0]})
    assert a[0] == DEFAULT_MAPPING["spread"]
    assert b[0] == DEFAULT_MAPPING["concentration"]


def test_zscore_handles_constant_column_without_nan():
    z = zscore([2.0, 2.0, 2.0])
    assert np.all(z == 0.0) and not np.isnan(z).any()


def test_constant_features_route_to_fallback_not_nan():
    feats = {"spread": [1, 1], "concentration": [1, 1], "logit_dispersion": [1, 1]}
    assert list(route(feats)) == [DEFAULT_FALLBACK, DEFAULT_FALLBACK]


def test_missing_feature_names_the_fix():
    with pytest.raises(KeyError, match="concentration"):
        route({"spread": [1.0], "logit_dispersion": [1.0]})


def test_mismatched_lengths_rejected():
    with pytest.raises(ValueError, match="mismatched"):
        route({"spread": [1, 2], "concentration": [1], "logit_dispersion": [1, 2]})


def test_route_scores_are_zscored_per_feature():
    z = route_scores({"spread": [0, 10], "concentration": [1, 1], "logit_dispersion": [-5, 5]})
    assert set(z) == set(DEFAULT_MAPPING)
    for v in z.values():
        assert abs(float(np.mean(v))) < 1e-9
