"""Module for the EWFS behaviors."""

from enum import StrEnum
from qiskit.providers.fake_provider import GenericBackendV2


class Behavior(StrEnum):
    """Supported behaviors for the friends."""

    CNOT_LADDER = "cnot_ladder"
    GHZ = "ghz"


CNOT_LADDER, GHZ = Behavior.CNOT_LADDER.value, Behavior.GHZ.value
BEHAVIORS = [CNOT_LADDER, GHZ]

# Backend for the GHZ circuit.
COUPLING_MAP = list(GenericBackendV2(num_qubits=156).coupling_map)
