"""Local friendliness Bell-like iinequalities for EWFS."""

from ewfs.facets import Facets


def compute_inequalities(results: dict, verbose: bool = False) -> dict:
    """Compute the LF inequalities."""
    facets = Facets(results)
    semi_brukner = facets.semi_brukner

    if verbose:
        print(f"{semi_brukner=} -- is violated: {semi_brukner > 0}")

    return {"semi_brukner": semi_brukner}


def compute_violations(
    results: dict, charlie_size: int, debbie_size: int, strategy: str, verbose: bool = False
) -> dict:
    """Compute violation values based on strategy."""
    if strategy == "random":
        return compute_inequalities(results=results, verbose=verbose)
    elif strategy == "majority_vote":
        return compute_inequalities(
            decode_results(results=results, charlie_size=charlie_size, debbie_size=debbie_size), verbose=verbose
        )
    raise ValueError(f"Strategy: {strategy} not defined.")


def decode_results(results: dict, charlie_size: int, debbie_size: int = 1) -> dict:
    """Take majority vote of measurement bit-strings."""
    decoded_results = {}

    # For each setting, there is a dictionary of measurement results.
    for setting in results:
        if setting == ("peek", "reverse_1") or setting == ("peek", "reverse_2"):
            # Debbie's size is 1 because no PEEK setting
            debbie_size = 1

            setting_results: dict = {}
            # Decode the keys for each measurement result of the setting.
            for k, v in results[setting].items():
                alice_friend, bob_friend = k[:charlie_size], k[-debbie_size:]

                alice_zero_count, bob_zero_count = alice_friend.count("0"), bob_friend.count("0")

                alice_decoding = "0" if alice_zero_count >= charlie_size // 2 + 1 else "1"
                bob_decoding = "0" if bob_zero_count >= 1 else "1"

                if alice_decoding + bob_decoding in setting_results.keys():
                    setting_results[alice_decoding + bob_decoding] += v
                else:
                    setting_results[alice_decoding + bob_decoding] = v
            decoded_results[setting] = setting_results
        else:
            decoded_results[setting] = results[setting]

    return decoded_results


def calculate_branch_factor(friend_size: int) -> float:
    """Branch factor is defined in arXiv:2106.16044v1 as the number of friends minus one."""
    assert friend_size > 0, "Friend size must be a positive integer."
    return friend_size - 1
