"""Module for the EWFS strategies."""

from enum import StrEnum


class Strategy(StrEnum):
    """Supported strategies for the super-observers."""

    MAJORITY_VOTE = "majority_vote"
    RANDOM = "random"


MAJORITY_VOTE, RANDOM = Strategy.MAJORITY_VOTE.value, Strategy.RANDOM.value
STRATEGIES = [MAJORITY_VOTE, RANDOM]
