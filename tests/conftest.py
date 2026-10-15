"""Shared pytest fixtures and configuration for the CEG test suite."""

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _seed_everything():
    """Seed the RNGs before every test for reproducibility.

    Individual tests may re-seed as needed; this only guarantees a
    deterministic starting point so unseeded ``torch.randn`` calls do not make
    assertions flaky across runs.
    """
    torch.manual_seed(0)
    np.random.seed(0)
    yield
