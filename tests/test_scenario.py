"""Tests for EWFS scenario."""

import pytest
from ewfs.scenario import EWFS, SETTINGS, STRATEGIES, DEFAULT_ANGLES, DEFAULT_BETA


@pytest.mark.parametrize("alice_setting", SETTINGS)
@pytest.mark.parametrize("bob_setting", SETTINGS)
@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("charlie_size", [1, 2])
@pytest.mark.parametrize("debbie_size", [1, 2])
def test_ewfs_initialization(alice_setting, bob_setting, strategy, charlie_size, debbie_size):
    """Test initialization of the EWFS class."""
    ewfs = EWFS(
        alice_setting=alice_setting,
        bob_setting=bob_setting,
        strategy=strategy,
        charlie_size=charlie_size,
        debbie_size=debbie_size,
    )

    assert ewfs.alice_setting == alice_setting
    assert ewfs.bob_setting == bob_setting
    assert ewfs.strategy == strategy
    assert ewfs.charlie_size == charlie_size
    assert ewfs.debbie_size == debbie_size
    assert ewfs.angles == DEFAULT_ANGLES
    assert ewfs.beta == DEFAULT_BETA
