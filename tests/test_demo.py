"""The shipped demo must run end-to-end with numpy alone (no torch, no vLLM, no GPU)."""
import numpy as np
import pytest

from vllm_logits import demo


@pytest.fixture(scope="module")
def data():
    return demo.load()


def test_shipped_data_loads_with_expected_schema(data):
    for col in ["cell", "problem_id", "retry", *demo.INTERVENTIONS,
                "spread", "concentration", "logit_dispersion"]:
        assert col in data, col
    n = len(data["retry"])
    assert n == 1423
    assert all(len(v) == n for v in data.values())


def test_rates_are_probabilities(data):
    for col in ["retry", *demo.INTERVENTIONS]:
        assert data[col].min() >= 0.0 and data[col].max() <= 1.0, col


def test_classify_partitions_every_failure_exactly_once(data):
    labels = demo.classify(data)
    assert set(labels) == {"resample", "steerable", "beyond reach"}
    assert len(labels) == len(data["retry"])


def test_beyond_reach_means_no_intervention_rescued(data):
    labels = demo.classify(data)
    best = np.max([data[o] for o in demo.INTERVENTIONS], axis=0)
    assert np.all(best[labels == "beyond reach"] == 0.0)


def test_steerable_obeys_the_stated_thresholds(data):
    labels = demo.classify(data)
    m = labels == "steerable"
    best = np.max([data[o] for o in demo.INTERVENTIONS], axis=0)
    assert np.all((best - data["retry"])[m] >= demo.TAU_STEERABLE)
    assert np.all((1.0 - data["retry"])[m] >= demo.TAU_HARD)


def test_flagship_cell_is_present_and_named_consistently(data):
    assert demo.FLAGSHIP in set(data["cell"])


def test_default_run_completes(capsys):
    demo.main([])
    out = capsys.readouterr().out
    assert "steerable" in out and "routed per failure" in out


def test_per_cell_run_completes_for_every_shipped_cell(data, capsys):
    for cell in sorted(set(data["cell"])):
        demo.main(["--cell", cell])
    assert "routed per failure" in capsys.readouterr().out


def test_unknown_cell_lists_the_valid_ones():
    with pytest.raises(SystemExit, match="available"):
        demo.main(["--cell", "nope|nope"])
