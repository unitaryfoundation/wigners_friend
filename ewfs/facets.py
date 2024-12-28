"""Facets of the EWFS scenario."""

from dataclasses import dataclass
from ewfs.utils import decode_results


@dataclass
class Facets:
    results: dict[tuple[str, str], dict]

    def compute_violations(self, charlie_size: int, debbie_size: int, strategy: str, verbose: bool = False) -> dict:
        """Compute violation values based on strategy."""
        if strategy == "majority_vote":
            self.results = decode_results(results=self.results, charlie_size=charlie_size, debbie_size=debbie_size)

        facets = Facets(self.results)
        semi_brukner = facets.semi_brukner

        if verbose:
            print(f"{semi_brukner=} -- is violated: {semi_brukner > 0}")

        return {"semi_brukner": semi_brukner}

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
