"""
Synthetic data attribution experiments.
"""

import json
import os
import random
import time
from typing import Any

import numpy as np
import torch
from captum.attr import IntegratedGradients

from causalig.attr import CausalIntegratedGradients
from causalig.attr._core.causal_expected_gradients import (
    CausalExpectedGradientsMultiStep,
)
from causalig.attr._core.causal_shap import CausalSHAP
from causalig.attr._core.conditional_shap import ConditionalSHAP
from causalig.models.surrogate import (
    MaskedGlobalLinear,
    create_autoencoder_from_scm,
    train_surrogate_model,
)
from experiments import BaseExperiment
from experiments.config.experiment import SyntheticExperimentConfig
from experiments.data.synthetic import compute_cate, generate_synthetic_data
from experiments.evaluation.metrics import compare_to_cate, evaluate_model_performance
from experiments.models.lasso import LassoRegression, train_lasso
from experiments.utils.causal import process_causal_features
from experiments.utils.graph_perturbation import (
    PerturbationMethod,
    perturb_graph,
    perturbation_filename_suffix,
)
from experiments.visualization.plots import plot_cate_comparison


class AttributionExperiment(BaseExperiment):
    """
    Shared pipeline of the synthetic benchmark.

    Generates synthetic data with known causal structure, trains a Lasso
    regression model and computes attributions. Evaluation against the CATE
    ground truth lives in `CausalAttributionExperiment`, which is the class
    to instantiate.
    """

    def __init__(self, config: SyntheticExperimentConfig):
        """
        Initialize the attribution experiment.

        Args:
            config: Configuration for the experiment
        """
        super().__init__(config)
        self.config: SyntheticExperimentConfig = config
        os.makedirs(self.config.output_dir, exist_ok=True)

    def generate_data(self) -> dict[str, Any]:
        """Generate synthetic data."""
        self.logger.info(
            f"Generating synthetic data: {self.config.num_features} features, "
            f"{self.config.num_causal} causal, {self.config.num_samples} samples"
        )

        (
            features,
            target,
            dag,
            causal_indices,
            scm,
            assignment_metadata,
            feature_mean,
            feature_std,
        ) = generate_synthetic_data(
            num_features=self.config.num_features,
            num_causal=self.config.num_causal,
            num_samples=self.config.num_samples_train,
            noise_multiplier=self.config.noise_multiplier,
            # Same default as `run()`: without it, sempler draws unseeded data.
            seed=0 if self.config.seed is None else self.config.seed,
            max_depth=self.config.max_depth,
            return_assignment_metadata=True,
        )

        # Convert to tensors
        features_tensor = torch.tensor(
            features, dtype=torch.float32, device=self.config.device
        )
        target_tensor = torch.tensor(
            target, dtype=torch.float32, device=self.config.device
        )
        dag_tensor = torch.tensor(dag, dtype=torch.float32, device=self.config.device)

        # The graph given to the attribution methods may be a corrupted
        # version of the true DAG, to test robustness to missing/incomplete
        # causal structure knowledge. Data generation and CATE ground truth
        # always use the true `dag_tensor`.
        observed_dag_tensor = _perturb_dag(
            dag_tensor,
            self.config.graph_perturbation_method,
            self.config.edge_drop_frac,
            seed=0 if self.config.seed is None else self.config.seed,
        )
        if self.config.edge_drop_frac > 0:
            self.logger.info(
                f"Applied {self.config.graph_perturbation_method!r} perturbation "
                f"({self.config.edge_drop_frac:.0%}) to the graph given to the "
                f"attribution methods: {int(dag_tensor.sum().item())} -> "
                f"{int(observed_dag_tensor.sum().item())} edges."
            )

        return {
            "features": features,
            "target": target,
            "features_tensor": features_tensor,
            "target_tensor": target_tensor,
            "dag_tensor": dag_tensor,
            "observed_dag_tensor": observed_dag_tensor,
            "causal_indices": causal_indices,
            "scm": scm,
            "assignment_metadata": assignment_metadata,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
        }

    def train_model(self, data: dict[str, Any]) -> LassoRegression:
        """Train Lasso regression model."""
        self.logger.info("Training Lasso regression model...")

        model = LassoRegression(
            n_features=self.config.num_features, lam=self.config.lasso_lambda
        ).to(self.config.device)

        model = train_lasso(
            model,
            data["features_tensor"],
            data["target_tensor"],
            lr=self.config.lasso_lr,
            steps=self.config.lasso_steps,
        )

        # Evaluate model performance
        perf = evaluate_model_performance(
            model,
            data["features_tensor"],
            data["target_tensor"],
            data["causal_indices"],
        )

        self.logger.info(f"Lasso Regression MSE: {perf['mse']:.4f}")
        self.logger.info(f"Non-zero weights: {perf['num_nonzero_weights']}")
        self.logger.info(
            f"Causal features learned: {perf['num_causal_learned']} out of {perf['total_causal']}"
        )
        self.logger.info(f"Learned weights: {perf['learned_weights']}")
        self.logger.info(f"Causal indices: {data['causal_indices']}")

        return model

    def _prepare_surrogate(self, data: dict[str, Any]) -> Any:
        """Prepare the surrogate model used for attribution."""
        if self.config.sim_autoencoder:
            # Use the true SCM as the surrogate
            self.logger.info("Using SCM as surrogate")
            surrogate = create_autoencoder_from_scm(
                data["scm"], structural=self.config.structural
            )
        else:
            surrogate = self._prepare_masked_global_linear(data)

        return surrogate

    def _prepare_masked_global_linear(self, data: dict[str, Any]) -> Any:
        """
        Train a `MaskedGlobalLinear` surrogate on the simulated data.

        Unlike `sim_autoencoder=True` (which hands CIG/CEG the true SCM as
        the surrogate), this fits the per-edge mechanism from
        `data["features"]` while still relying on the (possibly
        edge-dropped) graph structure in `observed_dag_tensor` as prior
        knowledge — mirroring how the method is used on real data, where
        only the graph is known and the mechanism must be learned.
        """
        self.logger.info("Training MaskedGlobalLinear surrogate...")

        features_tensor = torch.tensor(data["features"], dtype=torch.float32)

        surrogate = MaskedGlobalLinear(
            adjacency_matrix=data["observed_dag_tensor"],
            use_basis_expansion=self.config.surrogate_use_basis_expansion,
            X_train=features_tensor,
        )

        surrogate = train_surrogate_model(
            surrogate=surrogate,
            X=features_tensor,
            epochs=self.config.surrogate_epochs,
            batch_size=self.config.surrogate_batch_size,
            device=self.config.device,
            seed=0 if self.config.seed is None else self.config.seed,
            save_dir=self.config.output_dir,
            logger=self.logger,
            lr=self.config.surrogate_lr,
            weight_decay=self.config.surrogate_weight_decay,
            grad_clip_norm=1.0,
        )

        return surrogate

    def compute_attributions(
        self,
        model: LassoRegression,
        data: dict[str, Any],
        names: tuple[str, ...] = ("CIG", "IG"),
    ) -> dict[str, torch.Tensor]:
        """Compute attributions using CIG and IG."""
        self.logger.info("Computing attributions...")
        attribution_timings = {}
        completeness_delta = {}
        completeness_delta_samples = {}

        t_prepare_surrogate = _now()

        # Prepare surrogate
        surrogate = self._prepare_surrogate(data)
        attribution_timings["prepare_surrogate"] = _elapsed(t_prepare_surrogate)

        # MaskedGlobalLinear exposes an exact closed-form input Jacobian;
        # the SCM wrapper (sim_autoencoder=True) relies on autograd instead.
        has_closed_form_jacobian = isinstance(surrogate, MaskedGlobalLinear)

        # Get baseline
        baseline = self.config.attribution_baseline

        attributions = {}

        # Use a subset for attribution if needed
        data_to_explain = data["features_tensor"][: self.config.num_samples]

        if "CIG" in names:
            # Log time taken for CIG attribution
            t0 = _now()
            # CIG attributions
            self.logger.info("  Computing CIG attributions...")
            cig_explainer = CausalIntegratedGradients(model)
            cig_attributions, cig_delta = cig_explainer.attribute(
                data_to_explain,
                surrogate=surrogate,
                ordering=self.config.ordering,
                baseline=baseline,
                n_steps=self.config.attribution_n_steps,
                adjacency_matrix=data["observed_dag_tensor"],
                step_chunk_size=self.config.attribution_step_chunk_size,
                dim_chunk_size=self.config.attribution_dim_chunk_size,
                has_closed_form_jacobian=has_closed_form_jacobian,
                return_convergence_delta=True,
            )
            dt = _elapsed(t0)
            self.logger.info(f"  CIG attributions computed in {dt:.2f} seconds")
            attributions["CIG"] = cig_attributions
            attribution_timings["CIG"] = dt
            completeness_delta["CIG"] = _relative_completeness_delta(
                cig_attributions, cig_delta
            )
            completeness_delta_samples["CIG"] = _per_sample_completeness_delta(
                cig_attributions, cig_delta
            )
            self.logger.debug(f"  CIG attributions shape: {cig_attributions.shape}")

        if "CEG" in names:
            t0 = _now()
            self.logger.info("  Computing CEG attributions...")
            ceg_explainer = CausalExpectedGradientsMultiStep(model)
            n_baselines = self.config.attribution_n_baselines
            num_available = data["features_tensor"].size(0)
            if n_baselines > num_available:
                n_baselines = num_available

            baseline_indices = torch.randperm(num_available, device=self.config.device)[
                :n_baselines
            ]
            baselines = data["features_tensor"][baseline_indices]

            ceg_attributions, ceg_delta = ceg_explainer.attribute(
                data_to_explain,
                surrogate=surrogate,
                ordering=self.config.ordering,
                baselines=baselines,
                n_steps=int(
                    self.config.attribution_n_steps / n_baselines
                ),  # Divide steps by number of baselines to keep total samples constant
                adjacency_matrix=data["observed_dag_tensor"],
                step_chunk_size=self.config.attribution_step_chunk_size,
                dim_chunk_size=self.config.attribution_dim_chunk_size,
                has_closed_form_jacobian=has_closed_form_jacobian,
                return_convergence_delta=True,
            )
            dt = _elapsed(t0)
            self.logger.info(f"  CEG attributions computed in {dt:.2f} seconds")
            attributions["CEG"] = ceg_attributions
            attribution_timings["CEG"] = dt
            completeness_delta["CEG"] = _relative_completeness_delta(
                ceg_attributions, ceg_delta
            )
            completeness_delta_samples["CEG"] = _per_sample_completeness_delta(
                ceg_attributions, ceg_delta
            )
            self.logger.debug(f"  CEG attributions shape: {ceg_attributions.shape}")

        if "IG" in names:
            t0 = _now()
            # IG attributions
            self.logger.info("  Computing IG attributions...")
            ig_explainer = IntegratedGradients(model)
            ig_attributions, ig_delta = ig_explainer.attribute(
                data_to_explain,
                baselines=baseline.unsqueeze(0) if baseline.dim() == 1 else baseline,
                return_convergence_delta=True,
            )
            dt = _elapsed(t0)
            self.logger.info(f"  IG attributions computed in {dt:.2f} seconds")
            attributions["IG"] = ig_attributions
            attribution_timings["IG"] = dt
            completeness_delta["IG"] = _relative_completeness_delta(
                ig_attributions, ig_delta
            )
            completeness_delta_samples["IG"] = _per_sample_completeness_delta(
                ig_attributions, ig_delta
            )
            self.logger.debug(f"  IG attributions shape: {ig_attributions.shape}")

        if "CondSHAP" in names:
            t0 = _now()
            self.logger.info("  Computing Conditional SHAP attributions...")
            cond_shap_explainer = ConditionalSHAP(
                model,
                background=data["features"],
                n_samples=self.config.conditional_shap_n_samples,
                npermutations=self.config.conditional_shap_n_permutations,
                seed=0 if self.config.seed is None else self.config.seed,
            )
            cond_shap_attributions = cond_shap_explainer.attribute(data_to_explain)
            attributions["CondSHAP"] = torch.tensor(
                cond_shap_attributions, device=self.config.device
            )
            dt = _elapsed(t0)
            self.logger.info(
                f"  Conditional SHAP attributions computed in {dt:.2f} seconds"
            )
            attribution_timings["CondSHAP"] = dt
            self.logger.debug(
                f"  Conditional SHAP attributions shape: {cond_shap_attributions.shape}"
            )

        if "CausalSHAP" in names and "scm" in data:
            t0 = _now()
            self.logger.info("  Computing Causal SHAP attributions...")
            causal_shap_explainer = CausalSHAP(
                model,
                data["scm"],
                background=data["features"],
                n_samples=self.config.causal_shap_n_samples,
                npermutations=self.config.causal_shap_n_permutations,
                noise_sigma=self.config.causal_shap_noise_sigma,
                seed=0 if self.config.seed is None else self.config.seed,
            )
            causal_shap_attributions = causal_shap_explainer.attribute(data_to_explain)
            attributions["CausalSHAP"] = torch.tensor(
                causal_shap_attributions, device=self.config.device
            )
            dt = _elapsed(t0)
            self.logger.info(f"  Causal SHAP attributions computed in {dt:.2f} seconds")
            attribution_timings["CausalSHAP"] = dt
            self.logger.debug(
                f"  Causal SHAP attributions shape: {causal_shap_attributions.shape}"
            )

        if "timings" not in self.results:
            self.results["timings"] = {}
        self.results["timings"]["attributions_seconds"] = attribution_timings
        self.results["timings"]["completeness_delta"] = completeness_delta
        self.results["completeness_delta_samples"] = completeness_delta_samples

        return attributions

    def run(self, names=("CIG", "IG")) -> dict[str, Any]:
        """
        Run the complete experiment pipeline.

        Returns:
            Dictionary containing all experiment results
        """
        self.logger.info("=" * 60)
        self.logger.info(f"Starting {self.__class__.__name__}")
        self.logger.info("=" * 60)

        # Reset run-local results to avoid leaking state across repeated runs.
        self.results = {}

        seed = 0 if self.config.seed is None else self.config.seed
        self.logger.info(f"Using deterministic seed: {seed}")
        set_determinism(seed)
        self.results["timings"] = {}
        t_experiment = _now()

        try:
            # Generate data
            self.logger.info("Step 1/5: Generating data...")
            t_step = _now()
            data = self.generate_data()
            self.results["timings"]["generate_data_seconds"] = _elapsed(t_step)
            self.results["data"] = data

            # Train model
            self.logger.info("Step 2/5: Training model...")
            t_step = _now()
            model = self.train_model(data)
            self.results["timings"]["train_model_seconds"] = _elapsed(t_step)
            self.results["model"] = model

            # Compute attributions
            self.logger.info("Step 3/5: Computing attributions...")
            t_step = _now()
            attributions = self.compute_attributions(model, data, names=names)
            self.results["timings"]["compute_attributions_seconds"] = _elapsed(t_step)
            self.results["attributions"] = attributions

            # Evaluate
            self.logger.info("Step 4/5: Evaluating attributions...")
            t_step = _now()
            evaluation_results = self.evaluate(attributions, data)
            self.results["timings"]["evaluate_seconds"] = _elapsed(t_step)
            self.results["evaluation"] = evaluation_results

            # Visualize
            if hasattr(self.config, "save_plot") and self.config.save_plot:
                self.logger.info("Step 5/5: Visualizing results...")
                t_step = _now()
                fig = self.visualize(evaluation_results)
                self.results["timings"]["visualize_seconds"] = _elapsed(t_step)
                self.results["figure"] = fig

            self.results["timings"]["total_runtime_seconds"] = _elapsed(t_experiment)

            # Save results
            if hasattr(self.config, "save_results") and self.config.save_results:
                self.save_results(evaluation_results)

            self.logger.info("=" * 60)
            self.logger.info("Experiment completed successfully!")
            self.logger.info("=" * 60)

        except Exception as e:
            self.logger.error(f"Experiment failed with error: {e}", exc_info=True)
            raise

        return self.results


class CausalAttributionExperiment(AttributionExperiment):

    def compute_cate(self, model, data):

        self.logger.info("Computing CATE ground truth...")

        baseline = self.config.attribution_baseline
        # Use a subset for CATE computation if needed
        x = data["features_tensor"][: self.config.num_samples]
        scm = data["scm"]
        n_noise_samples = getattr(self.config, "cate_n_noise_samples", 10)
        self.logger.info(f"CATE Monte Carlo samples: {n_noise_samples}")

        t_cate = _now()
        cate = compute_cate(
            scm,
            model,
            x,
            baseline,
            n_noise_samples=n_noise_samples,
            seed=0 if self.config.seed is None else self.config.seed,
        )
        if "timings" not in self.results:
            self.results["timings"] = {}
        self.results["timings"]["compute_cate_seconds"] = _elapsed(t_cate)

        self.logger.info(f"CATE shape: {cate.shape}")
        return cate

    def evaluate(self, attributions, data, cate=None):
        """
        Evaluate attributions against the CATE ground truth.

        Args:
            cate: Precomputed CATE tensor to reuse (e.g. across an
                `edge_drop_frac` sweep, where the ground truth doesn't
                change). If ``None``, it is computed here as before.
        """

        data["direct_nodes"] = list(data["causal_indices"])

        data["eldest_ancestors"] = list(
            process_causal_features(
                data["causal_indices"],
                data["dag_tensor"],
                include_ancestors=True,
                only_eldest=True,
            )
        )

        data["all_ancestors"] = list(
            process_causal_features(
                data["causal_indices"],
                data["dag_tensor"],
                include_ancestors=True,
                only_eldest=False,
            )
        )

        data["all_nodes"] = list(range(self.config.num_features))

        self.logger.info("Running CAUSAL evaluation (CATE comparison)")
        self.logger.info(
            f"  Direct nodes (causal indices): {len(data['direct_nodes'])} features"
        )
        self.logger.info(
            f"  Eldest ancestors: {len(data['eldest_ancestors'])} features"
        )
        self.logger.info(f"  All ancestors: {len(data['all_ancestors'])} features")
        self.logger.info(f"  All nodes: {len(data['all_nodes'])} features")

        if cate is None:
            model = self.results["model"]
            cate = self.compute_cate(model, data)

        # Evaluate for each node set
        node_sets = {
            "direct_nodes": data["direct_nodes"],
            "eldest_ancestors": data["eldest_ancestors"],
            "all_nodes": data["all_nodes"],
        }

        results_by_node_set = {}

        for node_set_name, nodes_to_compare in node_sets.items():
            self.logger.info(
                f"\n--- Evaluating for {node_set_name} ({len(nodes_to_compare)} nodes) ---"
            )
            results = {}

            for name, attr in attributions.items():
                metrics = compare_to_cate(attr, cate, nodes_to_compare=nodes_to_compare)

                self.logger.info(
                    f"{name}: MSE={metrics['mse']:.6f}, Pearson={metrics['pearson']:.4f}, Spearman={metrics['spearman']:.4f}"
                )

                results[name] = metrics

            results_by_node_set[node_set_name] = results

        return {
            "mode": "causal",
            "cate": cate,
            "metrics_by_node_set": results_by_node_set,
            "attributions": attributions,
            "node_sets": {
                "direct_nodes": data["direct_nodes"],
                "all_ancestors": data["all_ancestors"],
                "eldest_ancestors": data["eldest_ancestors"],
            },
        }

    def visualize(
        self, evaluation_results: dict[str, Any], save_path: str | None = None
    ) -> Any:
        """Plot each method's attributions against the CATE ground truth."""
        figs = {}
        for name, attr in evaluation_results["attributions"].items():
            out_path = (
                f"{self.config.output_dir}/causal_comparison_{name}_"
                f"{self.config.ancestors_in_causal}_{self.config.only_eldest_ancestors}.png"
            )
            figs[name] = plot_cate_comparison(
                attr,
                evaluation_results["cate"],
                name,
                evaluation_results["node_sets"],
                save_path=out_path,
            )
            self.logger.debug(f"Visualization saved for {name}")

        return figs

    def save_results(self, evaluation_results, save_path=None):
        if save_path is None:
            save_path = f"{self.config.output_dir}/{self.config.get_results_filename()}"

        metrics_by_node_set = {
            node_set: {
                method: {metric: float(value) for metric, value in metric_dict.items()}
                for method, metric_dict in node_set_metrics.items()
            }
            for node_set, node_set_metrics in evaluation_results[
                "metrics_by_node_set"
            ].items()
        }

        save_dict = {
            "mode": "causal",
            "seed": self.config.seed,
            "ancestors_in_causal": self.config.ancestors_in_causal,
            "only_eldest_ancestors": self.config.only_eldest_ancestors,
            "sim_autoencoder": self.config.sim_autoencoder,
            "structural": self.config.structural,
            "graph_perturbation_method": self.config.graph_perturbation_method,
            "edge_drop_frac": self.config.edge_drop_frac,
            "metrics_by_node_set": metrics_by_node_set,
        }
        if "timings" in self.results:
            save_dict["timings_seconds"] = self.results["timings"]

        with open(save_path, "w") as f:
            json.dump(save_dict, f, indent=2)
        self.logger.info(f"Results saved to {save_path}")

    def run_multiple_seeds(
        self, seeds: list[int], names=("CIG", "IG")
    ) -> dict[str, Any]:
        """Run the causal experiment for multiple seeds and aggregate metrics."""
        if not seeds:
            raise ValueError("seeds must contain at least one integer seed")

        original_seed = self.config.seed
        original_save_results = getattr(self.config, "save_results", True)

        # Avoid overwriting per-seed files; persist only the final aggregate below.
        self.config.save_results = False

        per_seed_results = []

        try:
            for seed in seeds:
                self.logger.info("=" * 60)
                self.logger.info(f"Running seed {seed}")
                self.logger.info("=" * 60)
                self.config.seed = int(seed)

                run_results = self.run(names=names)
                evaluation = run_results.get("evaluation", {})

                if evaluation.get("mode") != "causal":
                    raise ValueError(
                        "run_multiple_seeds expects causal evaluation mode"
                    )

                metrics_by_node_set = {
                    node_set: {
                        method: {
                            metric: float(value)
                            for metric, value in metric_values.items()
                        }
                        for method, metric_values in node_set_metrics.items()
                    }
                    for node_set, node_set_metrics in evaluation[
                        "metrics_by_node_set"
                    ].items()
                }

                per_seed_results.append(
                    {
                        "seed": int(seed),
                        "metrics_by_node_set": metrics_by_node_set,
                        "timings": dict(run_results.get("timings", {})),
                        "completeness_delta_samples": run_results.get(
                            "completeness_delta_samples", {}
                        ),
                    }
                )

            # Aggregate mean/std per node_set, method and metric across seeds.
            aggregated_metrics_by_node_set = _aggregate_metrics_by_node_set(
                [seed_result["metrics_by_node_set"] for seed_result in per_seed_results]
            )

            summary = {
                "mode": "causal_multi_seed",
                "seeds": [int(s) for s in seeds],
                "num_seeds": len(seeds),
                "ancestors_in_causal": self.config.ancestors_in_causal,
                "only_eldest_ancestors": self.config.only_eldest_ancestors,
                "sim_autoencoder": self.config.sim_autoencoder,
                "structural": self.config.structural,
                "graph_perturbation_method": self.config.graph_perturbation_method,
                "edge_drop_frac": self.config.edge_drop_frac,
                "metrics_by_node_set": aggregated_metrics_by_node_set,
                "timings_seconds": _aggregate_timing_stats(per_seed_results),
                "per_seed": per_seed_results,
            }

            if original_save_results:
                save_path = (
                    f"{self.config.output_dir}/"
                    f"synthetic_causal_metrics_multiseed_{self.config.num_features}_"
                    f"noise{self.config.noise_multiplier}_"
                    f"ancestors{self.config.ancestors_in_causal}_"
                    f"eldest{self.config.only_eldest_ancestors}_"
                    f"sim_autoencoder{self.config.sim_autoencoder}_"
                    f"structural{self.config.structural}"
                    f"{self.config._edge_drop_suffix()}.json"
                )
                with open(save_path, "w") as f:
                    json.dump(summary, f, indent=2)
                self.logger.info(f"Multi-seed results saved to {save_path}")

            return summary

        finally:
            self.config.seed = original_seed
            self.config.save_results = original_save_results

    def run_edge_drop_sweep(
        self,
        edge_drop_fracs: list[float],
        seeds: list[int],
        names: tuple[str, ...] = ("CIG",),
        perturbation_method: str = "remove",
    ) -> dict[float, dict[str, Any]]:
        """
        Sweep multiple perturbation fractions for one graph-perturbation
        method, sharing data generation, model training, and CATE
        ground-truth computation across them for each seed — none of those
        three steps depend on the perturbation, only which graph is handed
        to the attribution methods does. Only attribution and evaluation are
        recomputed per fraction.

        Equivalent to calling `run_multiple_seeds` once per fraction, but
        without redundantly regenerating data / retraining the model /
        recomputing CATE (the CATE Monte Carlo estimate in particular is
        expensive, and scales with `num_features`).

        Args:
            edge_drop_fracs: Fractions (0-1) of `perturbation_method`'s
                eligible edges to perturb — see
                `experiments.utils.graph_perturbation` for what "eligible"
                means per method.
            seeds: Seeds to average over.
            names: Attribution methods to compute (typically just `("CIG",)`
                — the other methods either don't use the graph (IG) or are
                out of scope for this sweep).
            perturbation_method: One of
                `PerturbationMethod.{REMOVE, ADD_DAG, REVERSE}`'s values
                ("remove", "add_dag", "reverse"). NOTE: "add_dag" (and, for
                the same reason, "reverse") have no effect at all when
                `self.config.sim_autoencoder=True` — the true-SCM surrogate
                reads its own fixed parent columns and ignores anything
                extra a corrupted graph unmasks. Use a learned surrogate
                (`sim_autoencoder=False`) for those two
                methods to see any degradation. See
                `experiments.config.experiment.SyntheticExperimentConfig`'s
                `graph_perturbation_method` docstring for the full
                explanation.

        Returns:
            Mapping `edge_drop_frac -> multi-seed summary dict`, in the same
            format produced by `run_multiple_seeds`. One JSON file is saved
            per fraction (if `self.config.save_results`).
        """
        if not edge_drop_fracs:
            raise ValueError("edge_drop_fracs must contain at least one value")
        if not seeds:
            raise ValueError("seeds must contain at least one integer seed")

        original_seed = self.config.seed
        original_save_results = getattr(self.config, "save_results", True)

        per_seed_results_by_frac: dict[float, list[dict[str, Any]]] = {
            frac: [] for frac in edge_drop_fracs
        }

        try:
            for seed in seeds:
                self.logger.info("=" * 60)
                self.logger.info(
                    f"Running seed {seed} (data/model/CATE shared across "
                    f"{perturbation_method!r} fractions={edge_drop_fracs})"
                )
                self.logger.info("=" * 60)
                self.config.seed = int(seed)

                self.results = {}
                set_determinism(seed)
                self.results["timings"] = {}

                data = self.generate_data()
                model = self.train_model(data)
                self.results["model"] = model
                cate = self.compute_cate(model, data)

                for edge_drop_frac in edge_drop_fracs:
                    data["observed_dag_tensor"] = _perturb_dag(
                        data["dag_tensor"],
                        perturbation_method,
                        edge_drop_frac,
                        seed=seed,
                    )

                    attributions = self.compute_attributions(model, data, names=names)
                    evaluation = self.evaluate(attributions, data, cate=cate)

                    if evaluation.get("mode") != "causal":
                        raise ValueError(
                            "run_edge_drop_sweep expects causal evaluation mode"
                        )

                    metrics_by_node_set = {
                        node_set: {
                            attr_method: {
                                metric: float(value)
                                for metric, value in metric_values.items()
                            }
                            for attr_method, metric_values in node_set_metrics.items()
                        }
                        for node_set, node_set_metrics in evaluation[
                            "metrics_by_node_set"
                        ].items()
                    }

                    per_seed_results_by_frac[edge_drop_frac].append(
                        {
                            "seed": int(seed),
                            "metrics_by_node_set": metrics_by_node_set,
                            "timings": dict(self.results.get("timings", {})),
                            "completeness_delta_samples": self.results.get(
                                "completeness_delta_samples", {}
                            ),
                        }
                    )

            summaries = {}
            for edge_drop_frac in edge_drop_fracs:
                per_seed_results = per_seed_results_by_frac[edge_drop_frac]
                aggregated_metrics_by_node_set = _aggregate_metrics_by_node_set(
                    [r["metrics_by_node_set"] for r in per_seed_results]
                )

                summary = {
                    "mode": "causal_multi_seed",
                    "seeds": [int(s) for s in seeds],
                    "num_seeds": len(seeds),
                    "ancestors_in_causal": self.config.ancestors_in_causal,
                    "only_eldest_ancestors": self.config.only_eldest_ancestors,
                    "sim_autoencoder": self.config.sim_autoencoder,
                    "structural": self.config.structural,
                    "graph_perturbation_method": perturbation_method,
                    "edge_drop_frac": edge_drop_frac,
                    "metrics_by_node_set": aggregated_metrics_by_node_set,
                    "timings_seconds": _aggregate_timing_stats(per_seed_results),
                    "per_seed": per_seed_results,
                }
                summaries[edge_drop_frac] = summary

                if original_save_results:
                    edge_drop_suffix = perturbation_filename_suffix(
                        perturbation_method, edge_drop_frac
                    )
                    save_path = (
                        f"{self.config.output_dir}/"
                        f"synthetic_causal_metrics_multiseed_{self.config.num_features}_"
                        f"noise{self.config.noise_multiplier}_"
                        f"ancestors{self.config.ancestors_in_causal}_"
                        f"eldest{self.config.only_eldest_ancestors}_"
                        f"sim_autoencoder{self.config.sim_autoencoder}_"
                        f"structural{self.config.structural}"
                        f"{edge_drop_suffix}.json"
                    )
                    with open(save_path, "w") as f:
                        json.dump(summary, f, indent=2)
                    self.logger.info(
                        f"{perturbation_method!r} sweep results (frac={edge_drop_frac}) "
                        f"saved to {save_path}"
                    )

            return summaries

        finally:
            self.config.seed = original_seed
            self.config.save_results = original_save_results

    def run_nsteps_sweep(
        self,
        n_steps_list: list[int],
        seeds: list[int],
        names: tuple[str, ...] = ("CIG", "IG"),
    ) -> dict[int, dict[str, Any]]:
        """
        Sweep multiple `attribution_n_steps` values, sharing data generation,
        model training, and CATE ground-truth computation across them for
        each seed — none of those three steps depend on `n_steps`, only the
        CIG/CEG attribution integration does. Only attribution and
        evaluation are recomputed per `n_steps` value.

        Equivalent to calling `run_multiple_seeds` once per `n_steps` value
        (with `self.config.attribution_n_steps` set accordingly), but
        without redundantly regenerating data / retraining the model /
        recomputing CATE (the CATE Monte Carlo estimate in particular is
        expensive and scales with `num_features` — see
        `experiments.data.synthetic.compute_cate`'s per-feature loop).

        Args:
            n_steps_list: `attribution_n_steps` values to sweep.
            seeds: Seeds to average over.
            names: Attribution methods to compute.

        Returns:
            Mapping `n_steps -> multi-seed summary dict`, in the same format
            produced by `run_multiple_seeds` (plus an `attribution_n_steps`
            and `num_features` field for unambiguous loading). One JSON file
            is saved per `n_steps` value (if `self.config.save_results`),
            all under the same `output_dir`, distinguished by an
            `_nsteps<N>` filename suffix.
        """
        if not n_steps_list:
            raise ValueError("n_steps_list must contain at least one value")
        if not seeds:
            raise ValueError("seeds must contain at least one integer seed")

        original_seed = self.config.seed
        original_n_steps = self.config.attribution_n_steps
        original_save_results = getattr(self.config, "save_results", True)

        per_seed_results_by_nsteps: dict[int, list[dict[str, Any]]] = {
            n: [] for n in n_steps_list
        }

        try:
            for seed in seeds:
                self.logger.info("=" * 60)
                self.logger.info(
                    f"Running seed {seed} (data/model/CATE shared across "
                    f"n_steps={n_steps_list})"
                )
                self.logger.info("=" * 60)
                self.config.seed = int(seed)

                self.results = {}
                set_determinism(seed)
                self.results["timings"] = {}

                data = self.generate_data()
                model = self.train_model(data)
                self.results["model"] = model
                cate = self.compute_cate(model, data)

                for n_steps in n_steps_list:
                    self.config.attribution_n_steps = n_steps

                    attributions = self.compute_attributions(model, data, names=names)
                    evaluation = self.evaluate(attributions, data, cate=cate)

                    if evaluation.get("mode") != "causal":
                        raise ValueError(
                            "run_nsteps_sweep expects causal evaluation mode"
                        )

                    metrics_by_node_set = {
                        node_set: {
                            attr_method: {
                                metric: float(value)
                                for metric, value in metric_values.items()
                            }
                            for attr_method, metric_values in node_set_metrics.items()
                        }
                        for node_set, node_set_metrics in evaluation[
                            "metrics_by_node_set"
                        ].items()
                    }

                    per_seed_results_by_nsteps[n_steps].append(
                        {
                            "seed": int(seed),
                            "metrics_by_node_set": metrics_by_node_set,
                            "timings": dict(self.results.get("timings", {})),
                            "completeness_delta_samples": self.results.get(
                                "completeness_delta_samples", {}
                            ),
                        }
                    )

            summaries = {}
            for n_steps in n_steps_list:
                per_seed_results = per_seed_results_by_nsteps[n_steps]
                aggregated_metrics_by_node_set = _aggregate_metrics_by_node_set(
                    [r["metrics_by_node_set"] for r in per_seed_results]
                )

                summary = {
                    "mode": "causal_multi_seed",
                    "num_features": self.config.num_features,
                    "attribution_n_steps": n_steps,
                    "seeds": [int(s) for s in seeds],
                    "num_seeds": len(seeds),
                    "ancestors_in_causal": self.config.ancestors_in_causal,
                    "only_eldest_ancestors": self.config.only_eldest_ancestors,
                    "sim_autoencoder": self.config.sim_autoencoder,
                    "structural": self.config.structural,
                    "edge_drop_frac": self.config.edge_drop_frac,
                    "metrics_by_node_set": aggregated_metrics_by_node_set,
                    "timings_seconds": _aggregate_timing_stats(per_seed_results),
                    "per_seed": per_seed_results,
                }
                summaries[n_steps] = summary

                if original_save_results:
                    save_path = (
                        f"{self.config.output_dir}/"
                        f"synthetic_causal_metrics_multiseed_{self.config.num_features}_"
                        f"noise{self.config.noise_multiplier}_"
                        f"ancestors{self.config.ancestors_in_causal}_"
                        f"eldest{self.config.only_eldest_ancestors}_"
                        f"sim_autoencoder{self.config.sim_autoencoder}_"
                        f"structural{self.config.structural}"
                        f"_nsteps{n_steps}.json"
                    )
                    with open(save_path, "w") as f:
                        json.dump(summary, f, indent=2)
                    self.logger.info(f"n_steps={n_steps} results saved to {save_path}")

            return summaries

        finally:
            self.config.seed = original_seed
            self.config.attribution_n_steps = original_n_steps
            self.config.save_results = original_save_results

    def run_surrogate_comparison(
        self,
        surrogate_configs: dict[str, dict[str, Any]],
        seeds: list[int],
        names: tuple[str, ...] = ("CIG",),
    ) -> dict[str, Any]:
        """
        Compare attribution methods computed with different surrogates (e.g.
        the true SCM vs. a `MaskedGlobalLinear` fit to the simulated data),
        sharing data generation, model training, and CATE ground-truth
        computation across variants for each seed — none of those three
        steps depend on the surrogate, only which surrogate `compute_attributions`
        uses does.

        Args:
            surrogate_configs: Mapping from a label (used to suffix method
                names in the results, e.g. "oracle" -> "CIG_oracle") to a
                dict of `SyntheticExperimentConfig` field overrides applied
                before that variant's `compute_attributions` call, e.g.
                ``{"sim_autoencoder": True}`` or ``{"sim_autoencoder": False}``.
                Each dict may also include an ``"attribution_names"`` key
                (tuple of method names) to override `names` for that variant
                only — useful since `IG` doesn't depend on the surrogate and
                only needs to be computed once, e.g. include it alongside the
                cheap "oracle" variant rather than a "learned" variant that
                also has to train a surrogate.
            seeds: Seeds to average over.
            names: Default attribution method names computed for any variant
                that doesn't specify its own `"attribution_names"`.

        Returns:
            Multi-seed summary dict in the same format as `run_multiple_seeds`,
            with method names suffixed by their surrogate label (e.g.
            "CIG_oracle", "IG_oracle", "CIG_learned").
        """
        if not surrogate_configs:
            raise ValueError("surrogate_configs must contain at least one entry")
        if not seeds:
            raise ValueError("seeds must contain at least one integer seed")

        original_seed = self.config.seed
        original_save_results = getattr(self.config, "save_results", True)

        variant_overrides: dict[str, dict[str, Any]] = {}
        variant_names: dict[str, tuple[str, ...]] = {}
        touched_fields: set[str] = set()
        for label, overrides in surrogate_configs.items():
            overrides = dict(overrides)
            variant_names[label] = overrides.pop("attribution_names", names)
            variant_overrides[label] = overrides
            touched_fields.update(overrides)
        original_field_values = {
            field: getattr(self.config, field) for field in touched_fields
        }

        per_seed_results: list[dict[str, Any]] = []

        try:
            for seed in seeds:
                self.logger.info("=" * 60)
                self.logger.info(
                    f"Running seed {seed} (data/model/CATE shared across "
                    f"surrogate variants={list(surrogate_configs)})"
                )
                self.logger.info("=" * 60)
                self.config.seed = int(seed)

                self.results = {}
                set_determinism(seed)
                self.results["timings"] = {}

                data = self.generate_data()
                model = self.train_model(data)
                self.results["model"] = model
                cate = self.compute_cate(model, data)

                attributions = {}
                completeness_delta: dict[str, float] = {}
                completeness_delta_samples: dict[str, dict[str, list[float]]] = {}
                for label, overrides in variant_overrides.items():
                    for field, value in overrides.items():
                        setattr(self.config, field, value)

                    variant_attributions = self.compute_attributions(
                        model, data, names=variant_names[label]
                    )
                    variant_completeness_delta = self.results.get("timings", {}).get(
                        "completeness_delta", {}
                    )
                    variant_completeness_delta_samples = self.results.get(
                        "completeness_delta_samples", {}
                    )
                    for method_name, attr in variant_attributions.items():
                        suffixed = f"{method_name}_{label}"
                        attributions[suffixed] = attr
                        if method_name in variant_completeness_delta:
                            completeness_delta[suffixed] = variant_completeness_delta[
                                method_name
                            ]
                        if method_name in variant_completeness_delta_samples:
                            completeness_delta_samples[suffixed] = (
                                variant_completeness_delta_samples[method_name]
                            )

                evaluation = self.evaluate(attributions, data, cate=cate)

                if evaluation.get("mode") != "causal":
                    raise ValueError(
                        "run_surrogate_comparison expects causal evaluation mode"
                    )

                metrics_by_node_set = {
                    node_set: {
                        method: {
                            metric: float(value)
                            for metric, value in metric_values.items()
                        }
                        for method, metric_values in node_set_metrics.items()
                    }
                    for node_set, node_set_metrics in evaluation[
                        "metrics_by_node_set"
                    ].items()
                }

                per_seed_results.append(
                    {
                        "seed": int(seed),
                        "metrics_by_node_set": metrics_by_node_set,
                        "timings": {"completeness_delta": completeness_delta},
                        "completeness_delta_samples": completeness_delta_samples,
                    }
                )

            aggregated_metrics_by_node_set = _aggregate_metrics_by_node_set(
                [r["metrics_by_node_set"] for r in per_seed_results]
            )

            summary = {
                "mode": "causal_multi_seed",
                "seeds": [int(s) for s in seeds],
                "num_seeds": len(seeds),
                "ancestors_in_causal": self.config.ancestors_in_causal,
                "only_eldest_ancestors": self.config.only_eldest_ancestors,
                "surrogate_variants": variant_overrides,
                "metrics_by_node_set": aggregated_metrics_by_node_set,
                "timings_seconds": _aggregate_timing_stats(per_seed_results),
                "per_seed": per_seed_results,
            }

            if original_save_results:
                save_path = (
                    f"{self.config.output_dir}/"
                    f"synthetic_surrogate_comparison_{self.config.num_features}_"
                    f"noise{self.config.noise_multiplier}.json"
                )
                with open(save_path, "w") as f:
                    json.dump(summary, f, indent=2)
                self.logger.info(f"Surrogate comparison results saved to {save_path}")

            return summary

        finally:
            self.config.seed = original_seed
            self.config.save_results = original_save_results
            for field, value in original_field_values.items():
                setattr(self.config, field, value)


def _perturb_dag(
    dag_tensor: torch.Tensor, method: str, fraction: float, seed: int
) -> torch.Tensor:
    """Perturb `fraction` of `dag_tensor`'s edges via the shared graph-perturbation utility."""
    adjacency = dag_tensor.detach().cpu().numpy()
    perturbed, _ = perturb_graph(
        adjacency,
        method=PerturbationMethod(method),
        percentage=fraction * 100,
        seed=seed,
    )
    return torch.as_tensor(perturbed, dtype=dag_tensor.dtype, device=dag_tensor.device)


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _now() -> float:
    _sync_cuda()
    return time.perf_counter()


def _elapsed(start: float) -> float:
    _sync_cuda()
    return time.perf_counter() - start


def _relative_completeness_delta(
    attributions: torch.Tensor, delta: torch.Tensor
) -> float:
    """
    Batch-normalized completeness-axiom violation: total |delta| relative to
    the total model output difference the attributions are supposed to sum
    to, across the batch.

    Both causalig's and captum's `.attribute(..., return_convergence_delta=True)`
    return `delta = sum(attributions) - (f(input) - f(baseline))`, so the
    target quantity `f(input) - f(baseline)` can be recovered algebraically
    without a second forward pass.

    Normalizing at the batch level (sum of |delta| over sum of
    |expected_diff|) rather than averaging a per-sample ratio avoids the
    instability a per-sample relative error has whenever one sample's true
    effect `f(input) - f(baseline)` happens to be near zero — the same
    failure mode as MAPE in forecasting, where a single near-zero actual
    value can dominate the average even though its own error is otherwise
    unremarkable.
    """
    attribution_sum = attributions.sum(dim=-1)
    expected_diff = attribution_sum - delta
    return (delta.abs().sum() / (expected_diff.abs().sum() + 1e-8)).item()


def _per_sample_completeness_delta(
    attributions: torch.Tensor, delta: torch.Tensor
) -> dict[str, list[float]]:
    """
    Raw per-sample completeness-axiom values, for inspecting how well
    completeness holds across different explained samples within one batch
    — as opposed to `_relative_completeness_delta`'s single batch-aggregated
    WAPE scalar.

    Deliberately no ratio is taken here: `|delta_i|` and `|expected_diff_i|`
    are returned separately rather than divided, since a per-sample ratio is
    exactly the unstable quantity `_relative_completeness_delta` avoids (a
    single near-zero `expected_diff_i` would dominate it regardless of how
    unremarkable that sample's own `|delta_i|` is).
    """
    attribution_sum = attributions.sum(dim=-1)
    expected_diff = attribution_sum - delta
    return {
        "abs_delta": delta.abs().tolist(),
        "abs_expected_diff": expected_diff.abs().tolist(),
    }


def _aggregate_metrics_by_node_set(
    per_seed_metrics_by_node_set: list[dict[str, dict[str, dict[str, float]]]],
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    """Aggregate mean/std per node_set, method and metric across seeds."""
    node_sets = sorted(
        {node_set for metrics in per_seed_metrics_by_node_set for node_set in metrics}
    )

    aggregated_metrics_by_node_set = {}

    for node_set in node_sets:
        methods = sorted(
            {
                method
                for metrics in per_seed_metrics_by_node_set
                for method in metrics.get(node_set, {})
            }
        )
        aggregated_metrics = {}

        for method in methods:
            metric_names = sorted(
                {
                    metric
                    for metrics in per_seed_metrics_by_node_set
                    for metric in metrics.get(node_set, {}).get(method, {})
                }
            )
            aggregated_metrics[method] = {}

            for metric_name in metric_names:
                values = [
                    metrics[node_set][method][metric_name]
                    for metrics in per_seed_metrics_by_node_set
                    if node_set in metrics
                    and method in metrics[node_set]
                    and metric_name in metrics[node_set][method]
                ]

                values_arr = np.asarray(values, dtype=float)
                std = float(values_arr.std(ddof=1)) if len(values_arr) > 1 else 0.0
                aggregated_metrics[method][metric_name] = {
                    "mean": float(values_arr.mean()),
                    "std": std,
                    "values": values,
                }

        aggregated_metrics_by_node_set[node_set] = aggregated_metrics

    return aggregated_metrics_by_node_set


def _aggregate_timing_stats(per_seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate timing dicts across seeds (supports one-level nested timing dicts)."""
    collectors: dict[str, list[float]] = {}

    for seed_result in per_seed_results:
        timings = seed_result.get("timings", {})
        for key, value in timings.items():
            if isinstance(value, (int, float)):
                collectors.setdefault(key, []).append(float(value))
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        flat_key = f"{key}.{sub_key}"
                        collectors.setdefault(flat_key, []).append(float(sub_value))

    aggregated = {}
    for key, values in collectors.items():
        values_arr = np.asarray(values, dtype=float)
        aggregated[key] = {
            "mean": float(values_arr.mean()),
            "std": float(values_arr.std(ddof=1)) if len(values_arr) > 1 else 0.0,
            "values": values,
        }

    return aggregated


def set_determinism(seed: int) -> None:
    """Configure global RNG state and deterministic backends."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
