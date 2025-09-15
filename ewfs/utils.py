"""Local friendliness Bell-like iinequalities for EWFS."""

from ewfs.setting import PEEK, REVERSE_1, REVERSE_2


def decode_results(results: dict, charlie_size: int) -> dict:
    """Take majority vote of measurement bit-strings."""
    decoded_results = {}

    # For each setting, there is a dictionary of measurement results.
    for setting in results:
        if setting[0] == PEEK:
            # Debbie's size is 1 because no PEEK setting
            bob_size = 1

            setting_results: dict = {}
            # Decode the keys for each measurement result of the setting.
            for k, v in results[setting].items():
                alice_result, bob_result = k[-charlie_size:], k[:bob_size]

                alice_zero_count, bob_zero_count = alice_result.count("0"), bob_result.count("0")

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


def post_select_results(results: dict, flag_size: int = 0) -> dict:
    """ 
    Check the measurement outcomes of flag qubits and post-select results.
    """
    post_selected_results = {}

    for setting in results:
        post_selected_results_setting = {}

        for bitstring, count in results[setting].items():
            # Remove spaces
            processed_bitstring = bitstring.replace(" ", "")

            flags_bitstring = processed_bitstring[:flag_size]
            friends_bitstring = processed_bitstring[flag_size:]

            if flags_bitstring=="0"*flag_size:
                post_selected_results_setting[friends_bitstring] = count
    
        post_selected_results[setting] = post_selected_results_setting
        
            
    return post_selected_results


def calculate_branch_factor(friend_size: int) -> float:
    """Branch factor is defined in arXiv:2106.16044v1 as the number of friends minus one."""
    assert friend_size > 0, "Friend size must be a positive integer."
    return friend_size - 1
