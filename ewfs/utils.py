"""Local friendliness Bell-like iinequalities for EWFS."""

from ewfs.setting import PEEK, REVERSE_1, REVERSE_2


def decode_results(results: dict, charlie_size: int, debbie_size: int = 1) -> dict:
    """Take majority vote of measurement bit-strings."""
    decoded_results = {}

    # For each setting, there is a dictionary of measurement results.
    for setting in results:
        if setting == (PEEK, REVERSE_1) or setting == (PEEK, REVERSE_2):
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


def post_select_results(results: dict, charlie_size: int, debbie_size: int = 1) -> dict:
    """Check the measurement outcomes of flag qubits and post-select results."""
    # placeholder for post-selection logic
    return {}


def calculate_branch_factor(friend_size: int) -> float:
    """Branch factor is defined in arXiv:2106.16044v1 as the number of friends minus one."""
    assert friend_size > 0, "Friend size must be a positive integer."
    return friend_size - 1
