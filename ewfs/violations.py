"""Violations of the EWFS scenario."""

from ewfs.utils import decode_results, post_select_results
from ewfs.setting import SETTINGS, PEEK, REVERSE_1, REVERSE_2
from ewfs.strategy import MAJORITY_VOTE
from ewfs.observer import ALICE, BOB


def compute_violations(
    results: dict, charlie_size: int, debbie_size: int, strategy: str, friend_state: str, flag_size: int = 0, verbose: bool = False
) -> dict:
    """Compute violation values based on strategy."""
    if friend_state == "ghz":
        results = post_select_results(results, charlie_size, debbie_size, flag_size, 0) #debbie flag size hardcoded to 0

    if strategy is MAJORITY_VOTE:
        results = decode_results(results=results, charlie_size=charlie_size, debbie_size=debbie_size)

    A1B2 = double_expect((PEEK, REVERSE_1), results)
    A1B3 = double_expect((PEEK, REVERSE_2), results)
    A3B2 = double_expect((REVERSE_2, REVERSE_1), results)
    A3B3 = double_expect((REVERSE_2, REVERSE_2), results)

    semi_brukner = -A1B2 + A1B3 - A3B2 - A3B3 - 2

    if verbose:
        print(f"{semi_brukner=} -- is violated: {semi_brukner > 0}")
    return {"semi_brukner": semi_brukner}


def single_expect(observer: int, setting: str, results: dict) -> float:
    """Compute single expectation values for either Alice or Bob."""
    if observer is ALICE:
        ret = 0
        for settings in results.keys():
            if settings[ALICE] is setting:
                probs = results[settings]
                # <A> = P(00) + P(01) - P(10) - P(11)
                ret += probs.get("00", 0.0) + probs.get("01", 0.0) - probs.get("10", 0.0) - probs.get("11", 0.0)
        return ret / len(SETTINGS)
    else:
        ret = 0
        for settings in results.keys():
            if settings[BOB] is setting:
                probs = results[settings]
                # <B> = P(00) - P(01) + P(10) - P(11)
                ret += probs.get("00", 0.0) - probs.get("01", 0.0) + probs.get("10", 0.0) - probs.get("11", 0.0)
        return ret / len(SETTINGS)


def double_expect(settings: tuple[str, str], results: dict) -> float:
    """Expectation value of product of two operators."""
    probs = results[settings]
    # <AB> = P(00) - P(01) - P(10) + P(11)
    return probs.get("00", 0.0) - probs.get("01", 0.0) - probs.get("10", 0.0) + probs.get("11", 0.0)
