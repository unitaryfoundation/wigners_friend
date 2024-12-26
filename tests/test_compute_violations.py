"""Tests for EWFS violation computations."""

import pytest
import qiskit_aer
from ewfs.compute_violations import compute_violations


@pytest.mark.parametrize("backend", [qiskit_aer.Aer.get_backend("aer_simulator")])
@pytest.mark.parametrize("strategy", ["random", "majority_vote"])
@pytest.mark.parametrize("charlie_sizes, debbie_sizes", [(range(1, 2), range(1, 2)), (range(1, 3), range(1, 2))])
@pytest.mark.parametrize(
    "settings",
    [
        [
            ("peek", "reverse_1"),
            ("peek", "reverse_2"),
            ("reverse_2", "reverse_1"),
            ("reverse_2", "reverse_2"),
        ],
    ],
)
def test_compute_violations(backend, strategy, charlie_sizes, debbie_sizes, settings):
    results = compute_violations(
        shots=10_000,
        num_trials=1,
        charlie_sizes=charlie_sizes,
        debbie_sizes=debbie_sizes,
        strategy=strategy,
        backend=backend,
        settings=settings,
        verbose=False,
    )

    # Assert results are structured as expected
    assert isinstance(results, dict)
    for friend_size, violations in results.items():
        assert "semi_brukner" in violations
        for value in violations["semi_brukner"]:
            assert value > 0.75, f"Expected all semi_brukner values > 0.75 but got {value}"
