"""Facets of the EWFS scenario."""

from dataclasses import dataclass
from ewfs.utils import decode_results
from ewfs.scenario import ALICE, BOB, SETTINGS


@dataclass
class Facets:
    def __init__(self, results: dict, charlie_size: int, debbie_size: int, strategy: str) -> None:
        self.results = results
        self.charlie_size = charlie_size
        self.debbie_size = debbie_size
        self.strategy = strategy

    def compute_violations(self, verbose: bool = False) -> dict:
        """Compute violation values based on strategy."""
        if self.strategy == "majority_vote":
            self.results = decode_results(
                results=self.results, charlie_size=self.charlie_size, debbie_size=self.debbie_size
            )

        semi_brukner = self.semi_brukner
        brukner = self.brukner

        if verbose:
            print(f"{semi_brukner=} -- is violated: {semi_brukner > 0}")
            print(f"{brukner=} -- is violated: {brukner > 0}")
        return {"brukner": brukner, "semi_brukner": semi_brukner}

    def single_expect(self, observer: int, setting: str) -> float:
        """Compute single expectation values for either Alice or Bob."""
        if observer is ALICE:
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

    @property
    def brukner(self) -> float:
        """Calculate the Brukner facet as defined in Eq. (19) of arXiv:1907.05607."""
        A1B1 = self.double_expect(("peek", "peek"))
        A1B3 = self.double_expect(("peek", "reverse_2"))
        A2B1 = self.double_expect(("reverse_1", "peek"))
        A2B3 = self.double_expect(("reverse_1", "reverse_2"))

        return A1B1 - A1B3 - A2B1 - A2B3 - 2

    def facet_names(self) -> dict:
        """Retrieve all property names and their string representations."""
        property_strings = {}
        for attr_name in dir(self):
            attr = getattr(self.__class__, attr_name, None)
            if isinstance(attr, property):
                property_strings[attr_name] = getattr(self, attr_name)
        return property_strings
