"""Tests for :class:`experiments.config.experiment.SyntheticExperimentConfig`.

Focuses on default construction, the ``__post_init__`` validation
(``num_causal`` must not exceed ``num_features``, device normalization, default
baseline) and the filename helpers.
"""

import pytest
import torch

from experiments.config.experiment import SyntheticExperimentConfig


def test_default_construction_populates_derived_fields():
    cfg = SyntheticExperimentConfig()

    # A few of the documented defaults.
    assert cfg.num_features == 100
    assert cfg.num_causal == 20
    assert cfg.num_causal <= cfg.num_features

    # __post_init__ normalizes device to a torch.device...
    assert isinstance(cfg.device, torch.device)

    # ...and builds a zero baseline of length num_features on that device.
    assert cfg.attribution_baseline is not None
    assert cfg.attribution_baseline.shape == (cfg.num_features,)
    assert cfg.attribution_baseline.device.type == cfg.device.type
    assert torch.count_nonzero(cfg.attribution_baseline) == 0


def test_num_causal_exceeding_num_features_raises():
    with pytest.raises(ValueError):
        SyntheticExperimentConfig(num_features=10, num_causal=11)


def test_num_causal_equal_to_num_features_is_allowed():
    # The validation only rejects strictly greater; equality is fine.
    cfg = SyntheticExperimentConfig(num_features=8, num_causal=8, device="cpu")
    assert cfg.num_causal == cfg.num_features


def test_device_string_is_converted_and_baseline_sized_accordingly():
    cfg = SyntheticExperimentConfig(num_features=7, num_causal=3, device="cpu")

    assert isinstance(cfg.device, torch.device)
    assert cfg.device.type == "cpu"
    assert cfg.attribution_baseline.shape == (7,)
    assert cfg.attribution_baseline.device.type == "cpu"


def test_custom_baseline_moved_to_device():
    baseline = torch.arange(5, dtype=torch.float32)
    cfg = SyntheticExperimentConfig(
        num_features=5, num_causal=2, device="cpu", attribution_baseline=baseline
    )

    # A user-provided baseline is preserved (moved to the config device),
    # not overwritten with zeros.
    assert cfg.attribution_baseline.device.type == "cpu"
    torch.testing.assert_close(cfg.attribution_baseline, baseline.to("cpu"))


def test_filename_helpers_reflect_config():
    cfg = SyntheticExperimentConfig(
        num_features=12,
        num_causal=4,
        noise_multiplier=0.25,
        device="cpu",
    )

    results_name = cfg.get_results_filename()

    assert results_name.startswith("synthetic_causal_metrics_")
    assert results_name.endswith(".json")
    # num_features and noise multiplier are encoded into the filename.
    assert "12" in results_name and "noise0.25" in results_name


def test_filename_omits_perturbation_suffix_when_frac_is_zero():
    cfg = SyntheticExperimentConfig(device="cpu")
    assert "edgedrop" not in cfg.get_results_filename()
    assert "adddag" not in cfg.get_results_filename()
    assert "reverse" not in cfg.get_results_filename()


def test_filename_perturbation_suffix_keeps_legacy_token_for_remove():
    # "remove" (the only method that existed before add_dag/reverse) keeps
    # the original "edgedrop" token so already-generated result files stay
    # discoverable under the same naming convention.
    cfg = SyntheticExperimentConfig(
        device="cpu", graph_perturbation_method="remove", edge_drop_frac=0.5
    )
    assert "_edgedrop0.5" in cfg.get_results_filename()


def test_filename_perturbation_suffix_distinguishes_add_dag_and_reverse():
    add_cfg = SyntheticExperimentConfig(
        device="cpu", graph_perturbation_method="add_dag", edge_drop_frac=0.5
    )
    reverse_cfg = SyntheticExperimentConfig(
        device="cpu", graph_perturbation_method="reverse", edge_drop_frac=0.5
    )
    assert "_adddag0.5" in add_cfg.get_results_filename()
    assert "_reverse0.5" in reverse_cfg.get_results_filename()
    assert add_cfg.get_results_filename() != reverse_cfg.get_results_filename()
