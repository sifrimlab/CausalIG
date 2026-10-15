"""
Data generation and utilities module.
"""

from experiments.data.assignments import generate_random_assignments_torch_and_numpy
from experiments.data.synthetic import generate_synthetic_data

__all__ = ["generate_synthetic_data", "generate_random_assignments_torch_and_numpy"]
