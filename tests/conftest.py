import pytest
import numpy as np

# Ensure reproducibility across all tests
@pytest.fixture(autouse=True)
def set_seed():
    np.random.seed(42)
