"""Facets of the EWFS scenario."""

from dataclasses import dataclass
from ewfs.scenario import ALICE, BOB, SETTINGS


@dataclass
class Facets:
    results: dict[tuple[str, str], dict]

    def single_expect(self, observer: int, setting: tuple[str, str]) -> float:
        """Compute single expectation values for either Alice or Bob."""
        if observer == ALICE:
            ret = 0
            for settings in self.results.keys():
                if settings[ALICE] is setting:
                    probs = self.results[settings]
                    # <A> = P(00) + P(01) - P(10) - P(11)
                    ret += probs.get("00", 0.0) + probs.get("01", 0.0) - probs.get("10", 0.0) - probs.get("11", 0.0)
            return ret / len(SETTINGS)
        else:
            ret = 0
            for settings in self.results.keys():
                if settings[BOB] is setting:
                    probs = self.results[settings]
                    # <B> = P(00) - P(01) + P(10) - P(11)
                    ret += probs.get("00", 0.0) - probs.get("01", 0.0) + probs.get("10", 0.0) - probs.get("11", 0.0)
            return ret / len(SETTINGS)

    def double_expect(self, settings: tuple[str, str]) -> float:
        """Expectation value of product of two operators."""
        probs = self.results[settings]
        # <AB> = P(00) - P(01) - P(10) + P(11)
        return probs.get("00", 0.0) - probs.get("01", 0.0) - probs.get("10", 0.0) + probs.get("11", 0.0)

    @property
    def semi_brukner(self) -> float:
        """Calculate the semi-brukner facet as defined in Eq. (18) of arXiv:1907.05607."""
        A1B2 = self.double_expect(("peek", "reverse_1"))
        A1B3 = self.double_expect(("peek", "reverse_2"))
        A3B2 = self.double_expect(("reverse_2", "reverse_1"))
        A3B3 = self.double_expect(("reverse_2", "reverse_2"))
        return -A1B2 + A1B3 - A3B2 - A3B3 - 2
