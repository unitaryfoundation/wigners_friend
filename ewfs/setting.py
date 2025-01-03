"""Experiment settings and setting pairs for the EWFS protocol."""

from enum import StrEnum
import itertools


class Setting(StrEnum):
    """Experiment settings (peek, reverse_1, and reverse_2)."""

    PEEK = "peek"
    REVERSE_1 = "reverse_1"
    REVERSE_2 = "reverse_2"


PEEK, REVERSE_1, REVERSE_2 = Setting.PEEK.value, Setting.REVERSE_1.value, Setting.REVERSE_2.value
SETTINGS = [PEEK, REVERSE_1, REVERSE_2]

# All possible pairs of settings for Alice and Bob.
SETTING_PAIRS = [setting for setting in itertools.product(SETTINGS, repeat=2)]
