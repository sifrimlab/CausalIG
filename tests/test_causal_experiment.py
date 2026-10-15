import numpy as np
import pytest
import torch

from experiments.config.experiment import SyntheticExperimentConfig
from experiments.synthetic import CausalAttributionExperiment
from experiments.utils.graph_perturbation import is_dag


def _tiny_config(tmp_path, save_plot):
    num_features = 8
    return SyntheticExperimentConfig(
        num_features=num_features,
        num_causal=2,
        num_samples=4,
        num_samples_train=64,
        noise_multiplier=0.1,
        ancestors_in_causal=True,
        only_eldest_ancestors=True,
        sim_autoencoder=True,
        structural=True,
        ordering=list(np.arange(num_features)),
        device=torch.device("cpu"),
        max_depth=2,
        seed=0,
        attribution_n_steps=4,
        output_dir=str(tmp_path),
        log_dir=str(tmp_path),
        save_plot=save_plot,
        save_results=False,
    )


def test_causal_experiment_run_end_to_end(tmp_path):
    config = _tiny_config(tmp_path, save_plot=False)
    experiment = CausalAttributionExperiment(config)

    results = experiment.run(names=["IG"])

    evaluation = results["evaluation"]
    assert evaluation["mode"] == "causal"
    assert "IG" in evaluation["metrics_by_node_set"]["direct_nodes"]


def test_edge_drop_frac_hides_edges_from_attribution_methods(tmp_path):
    config = _tiny_config(tmp_path, save_plot=False)
    config.edge_drop_frac = 0.5
    experiment = CausalAttributionExperiment(config)

    results = experiment.run(names=["IG"])

    data = results["data"]
    true_edges = int(data["dag_tensor"].sum().item())
    observed_edges = int(data["observed_dag_tensor"].sum().item())

    # The true DAG (used for data generation and CATE ground truth) is
    # untouched; only the graph handed to the attribution methods is
    # corrupted.
    assert observed_edges < true_edges
    assert observed_edges == true_edges - round(0.5 * true_edges)


def test_graph_perturbation_method_add_dag_adds_edges(tmp_path):
    config = _tiny_config(tmp_path, save_plot=False)
    config.graph_perturbation_method = "add_dag"
    config.edge_drop_frac = 0.5
    experiment = CausalAttributionExperiment(config)

    results = experiment.run(names=["IG"])

    data = results["data"]
    true_edges = int(data["dag_tensor"].sum().item())
    observed_edges = int(data["observed_dag_tensor"].sum().item())

    assert observed_edges > true_edges
    assert is_dag(data["observed_dag_tensor"].cpu().numpy())


def test_graph_perturbation_method_reverse_keeps_edge_count(tmp_path):
    config = _tiny_config(tmp_path, save_plot=False)
    config.graph_perturbation_method = "reverse"
    config.edge_drop_frac = 0.5
    experiment = CausalAttributionExperiment(config)

    results = experiment.run(names=["IG"])

    data = results["data"]
    true_edges = int(data["dag_tensor"].sum().item())
    observed_edges = int(data["observed_dag_tensor"].sum().item())

    # Reversing flips direction, it never adds/removes edges.
    assert observed_edges == true_edges
    assert is_dag(data["observed_dag_tensor"].cpu().numpy())


def test_add_dag_has_no_effect_under_oracle_surrogate(tmp_path):
    """
    Regression test for a real finding, not a bug: with sim_autoencoder=True
    (the true SCM standing in as the surrogate), CIG's structural forward
    pass reads each node's true-parent columns straight off the SCM's own
    fixed adjacency, ignoring any extra columns a corrupted graph happens to
    unmask. So "add_dag" (which only adds edges, never removing a true
    parent) must leave CIG's attributions completely unchanged, however
    large the corruption. Only a learned surrogate (see the sibling test)
    can actually be misled by spurious edges.
    """
    config = _tiny_config(tmp_path, save_plot=False)
    assert config.sim_autoencoder is True
    experiment = CausalAttributionExperiment(config)

    data = experiment.generate_data()
    model = experiment.train_model(data)

    data["observed_dag_tensor"] = data["dag_tensor"].clone()
    attr_baseline = experiment.compute_attributions(model, data, names=["CIG"])[
        "CIG"
    ].clone()

    data["observed_dag_tensor"] = torch.ones_like(data["dag_tensor"]) - torch.eye(
        config.num_features
    )
    attr_almost_complete_graph = experiment.compute_attributions(
        model, data, names=["CIG"]
    )["CIG"].clone()

    assert torch.equal(attr_baseline, attr_almost_complete_graph)


def test_add_dag_does_affect_learned_surrogate(tmp_path):
    """Counterpart to the oracle test above: a learned MaskedGlobalLinear
    surrogate (trained on whatever graph it's given) has no privileged access
    to the true structure, so spurious edges from add_dag *do* change its
    attributions."""
    config = _tiny_config(tmp_path, save_plot=False)
    config.sim_autoencoder = False
    config.surrogate_epochs = 2
    config.surrogate_batch_size = 8
    experiment = CausalAttributionExperiment(config)

    data = experiment.generate_data()
    model = experiment.train_model(data)

    data["observed_dag_tensor"] = data["dag_tensor"].clone()
    attr_baseline = experiment.compute_attributions(model, data, names=["CIG"])[
        "CIG"
    ].clone()

    data["observed_dag_tensor"] = torch.ones_like(data["dag_tensor"]) - torch.eye(
        config.num_features
    )
    attr_almost_complete_graph = experiment.compute_attributions(
        model, data, names=["CIG"]
    )["CIG"].clone()

    assert not torch.equal(attr_baseline, attr_almost_complete_graph)


def test_run_edge_drop_sweep_supports_add_dag_and_reverse(tmp_path):
    for perturbation_method in ["add_dag", "reverse"]:
        method_dir = tmp_path / perturbation_method
        method_dir.mkdir()
        config = _tiny_config(method_dir, save_plot=False)
        experiment = CausalAttributionExperiment(config)

        summaries = experiment.run_edge_drop_sweep(
            edge_drop_fracs=[0.0, 0.5],
            seeds=[0, 1],
            names=["CIG"],
            perturbation_method=perturbation_method,
        )

        assert set(summaries.keys()) == {0.0, 0.5}
        for summary in summaries.values():
            assert summary["graph_perturbation_method"] == perturbation_method
            assert "CIG" in summary["metrics_by_node_set"]["eldest_ancestors"]


def test_run_edge_drop_sweep_matches_run_multiple_seeds_per_fraction(tmp_path):
    seeds = [0, 1]

    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    baseline_config = _tiny_config(baseline_dir, save_plot=False)
    baseline_config.edge_drop_frac = 0.5
    baseline = CausalAttributionExperiment(baseline_config).run_multiple_seeds(
        names=["CIG"], seeds=seeds
    )

    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    sweep_config = _tiny_config(sweep_dir, save_plot=False)
    swept = CausalAttributionExperiment(sweep_config).run_edge_drop_sweep(
        edge_drop_fracs=[0.0, 0.5], seeds=seeds, names=["CIG"]
    )

    assert set(swept.keys()) == {0.0, 0.5}

    baseline_metrics = baseline["metrics_by_node_set"]["eldest_ancestors"]["CIG"]
    swept_metrics = swept[0.5]["metrics_by_node_set"]["eldest_ancestors"]["CIG"]

    # Reusing generate_data/train_model/compute_cate across the sweep must
    # give the same numbers as rerunning the whole pipeline per fraction.
    assert swept_metrics["spearman"]["mean"] == pytest.approx(
        baseline_metrics["spearman"]["mean"]
    )
    assert swept_metrics["mse"]["mean"] == pytest.approx(
        baseline_metrics["mse"]["mean"]
    )


def test_run_edge_drop_sweep_rejects_empty_inputs(tmp_path):
    config = _tiny_config(tmp_path, save_plot=False)
    experiment = CausalAttributionExperiment(config)

    with pytest.raises(ValueError):
        experiment.run_edge_drop_sweep(edge_drop_fracs=[], seeds=[0])
    with pytest.raises(ValueError):
        experiment.run_edge_drop_sweep(edge_drop_fracs=[0.0], seeds=[])


def test_run_surrogate_comparison_compares_oracle_and_learned_surrogates(tmp_path):
    config = _tiny_config(tmp_path, save_plot=False)
    config.surrogate_batch_size = 8
    config.surrogate_epochs = 2
    experiment = CausalAttributionExperiment(config)

    results = experiment.run_surrogate_comparison(
        surrogate_configs={
            "oracle": {
                "sim_autoencoder": True,
                "attribution_names": ("CIG", "IG"),
            },
            "learned": {
                "sim_autoencoder": False,
                "attribution_names": ("CIG",),
            },
        },
        seeds=[0, 1],
    )

    methods = results["metrics_by_node_set"]["direct_nodes"]
    assert set(methods.keys()) == {"CIG_oracle", "IG_oracle", "CIG_learned"}

    # Config fields toggled per variant must be restored afterwards.
    assert config.sim_autoencoder is True


def test_run_surrogate_comparison_rejects_empty_inputs(tmp_path):
    config = _tiny_config(tmp_path, save_plot=False)
    experiment = CausalAttributionExperiment(config)

    with pytest.raises(ValueError):
        experiment.run_surrogate_comparison(surrogate_configs={}, seeds=[0])
    with pytest.raises(ValueError):
        experiment.run_surrogate_comparison(
            surrogate_configs={"oracle": {"sim_autoencoder": True}}, seeds=[]
        )


def test_causal_experiment_visualize_saves_plots(tmp_path):
    config = _tiny_config(tmp_path, save_plot=True)
    experiment = CausalAttributionExperiment(config)

    results = experiment.run(names=["IG"])

    assert "figure" in results
    out_path = (
        tmp_path
        / f"causal_comparison_IG_{config.ancestors_in_causal}_{config.only_eldest_ancestors}.png"
    )
    assert out_path.exists()
