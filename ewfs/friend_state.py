"""Module for the EWFS friend state."""

from enum import StrEnum


class FriendState(StrEnum):
    """Supported states for the friends."""

    CNOT_LADDER = "cnot_ladder"
    GHZ = "ghz"


CNOT_LADDER, GHZ = FriendState.CNOT_LADDER.value, FriendState.GHZ.value
FRIEND_STATES = [CNOT_LADDER, GHZ]
