"""Module for defining the super-observers (Alice and Bob) and their measurement angles."""

import numpy as np
from enum import Enum

from ewfs.setting import PEEK, REVERSE_1, REVERSE_2


class Observer(Enum):
    """ "Super"-observers (Alice and Bob)."""

    ALICE = 0
    BOB = 1


ALICE, BOB = Observer.ALICE.value, Observer.BOB.value


# (Optimized) angles and beta term used for Alice and Bob measurement operators. Adapted from arXiv:1907.05607. Note
# that despite the fact that degrees are used, we need to convert this to radians.
DEFAULT_ANGLES = {
    PEEK: np.deg2rad(40),
    REVERSE_1: np.deg2rad(230),
    REVERSE_2: np.deg2rad(310),
}
DEFAULT_BETA = np.deg2rad(220)
