"""Module for the EWFS friend state."""

from enum import StrEnum
from qiskit.providers.fake_provider import GenericBackendV2


class FriendState(StrEnum):
    """Supported states for the friends."""

    CNOT_LADDER = "cnot_ladder"
    GHZ = "ghz"


CNOT_LADDER, GHZ = FriendState.CNOT_LADDER.value, FriendState.GHZ.value
FRIEND_STATES = [CNOT_LADDER]

# Backend for the GHZ circuit.
COUPLING_MAP = list(GenericBackendV2(num_qubits=156).coupling_map)
