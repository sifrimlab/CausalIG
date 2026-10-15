"""
Experiment framework for attribution methods.

This module provides a base class for experiments that can be extended
to support different types of attribution experiments.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from causalig.utils.logging_utils import setup_experiment_logger


# Import base class first
class BaseExperiment(ABC):
    """
    Base class for attribution experiments.

    This class defines the interface that all experiment types should implement,
    allowing for extensibility to different experiment types (synthetic, real data, etc.).
    """

    def __init__(self, config: Any):
        """
        Initialize the experiment.

        Args:
            config: Configuration object for the experiment
        """
        self.config = config
        self.results: dict[str, Any] = {}

        # Set up logger if config has logging parameters
        if hasattr(config, "log_dir"):

            experiment_name = self.__class__.__name__
            self.logger = setup_experiment_logger(
                experiment_name=experiment_name,
                log_dir=config.log_dir,
                log_level=getattr(config, "log_level", logging.INFO),
                console_output=getattr(config, "log_to_console", True),
            )
        else:
            self.logger = logging.getLogger(self.__class__.__name__)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)

    def generate_data(self) -> dict[str, Any]:
        """
        Generate or load data for the experiment.

        Returns:
            Dictionary containing data components (features, targets, etc.)
        """
        pass

    def train_model(self, data: dict[str, Any]) -> Any:
        """
        Train the model to be explained.

        Args:
            data: Dictionary from generate_data()

        Returns:
            Trained model
        """
        pass

    @abstractmethod
    def compute_attributions(self, model: Any, data: dict[str, Any]) -> dict[str, Any]:
        """
        Compute attributions using different methods.

        Args:
            model: Trained model from train_model()
            data: Dictionary from generate_data()

        Returns:
            Dictionary mapping method names to attribution tensors
        """
        raise NotImplementedError("Subclasses must implement compute_attributions()")

    def evaluate(
        self, attributions: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Evaluate attribution quality.

        Args:
            attributions: Dictionary from compute_attributions()
            data: Dictionary from generate_data()

        Returns:
            Dictionary of evaluation metrics
        """
        pass

    def visualize(
        self, evaluation_results: dict[str, Any], save_path: str | None = None
    ) -> Any:
        """
        Visualize experiment results.

        Args:
            evaluation_results: Dictionary from evaluate()
            save_path: Optional path to save visualization

        Returns:
            Visualization object (e.g., matplotlib Figure)
        """
        # Default implementation does nothing
        # Subclasses can override
        return None

    def save_results(
        self, evaluation_results: dict[str, Any], save_path: str | None = None
    ) -> None:
        """
        Save experiment results to disk.

        Args:
            evaluation_results: Dictionary from evaluate()
            save_path: Optional path to save results
        """
        # Default implementation does nothing
        # Subclasses can override
        pass

    @abstractmethod
    def run(self):
        """
        Run the full experiment pipeline.
        """
        raise NotImplementedError("Subclasses must implement run()")


# Import concrete implementations (after BaseExperiment to avoid a circular import)
from experiments.synthetic import AttributionExperiment  # noqa: E402

__all__ = ["BaseExperiment", "AttributionExperiment"]
