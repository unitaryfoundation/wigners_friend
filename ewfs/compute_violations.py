"""Run EWFS experiment."""

from typing import Any
import itertools
from collections import defaultdict
import qiskit
import qiskit_aer

from ewfs.scenario import ALICE, BOB, SETTINGS, EWFS


DEFAULT_SETTINGS = [setting for setting in itertools.product(["peek", "reverse_1", "reverse_2"], repeat=2)]
EWFSExpectations = dict[tuple[str, str], dict[Any, Any]]


def compute_violations(
    shots: int = 10_000,
    num_trials: int = 2,
    charlie_sizes: range = range(1, 3),
    debbie_sizes: range = range(1, 2),
    strategy: str = "majority_vote",
    backend: qiskit_aer.backends.aerbackend.AerBackend = qiskit_aer.Aer.get_backend("aer_simulator"),
    settings: list[tuple] = DEFAULT_SETTINGS,
    verbose: bool = True,
) -> dict:
    """Compute EWFS violations for a range of friend sizes.

    Args:
        shots: Number of shots for each circuit.
        num_trials: Number of trials for each task.
        charlie_sizes: Range of Charlie's qubit sizes.
        debbie_sizes: Range of Debbie's qubit sizes.
        strategy: Strategy for the friend to use.
        backend: Backend to run the circuits.
        settings: List of Alice and Bob settings.
        verbose: Print intermediate results.

    Returns:
        post_processed_results: Post-processed results of the violations.
    """
    # The tasks dictionary has a key that corresponds to the qubit
    # system size with an associated value of the task for that size.
    tasks: defaultdict = defaultdict(dict)

    # Construct all circuits to be run over all qubit sizes.
    friend_sizes = []
    for charlie_size in charlie_sizes:
        for debbie_size in debbie_sizes:
            for trial in range(1, num_trials + 1):
                circuits = {}

                # Create circuits for each EWFS setting.
                # Note, we remove barriers in the Qiskit circuit.
                for alice_setting, bob_setting in settings:
                    circuit = EWFS(
                        alice_setting=alice_setting,
                        bob_setting=bob_setting,
                        strategy=strategy,
                        charlie_size=charlie_size,
                        debbie_size=debbie_size,
                    ).circuit()
                    circuits[(alice_setting, bob_setting)] = circuit

                # Use backend to transpile circuits.
                transpiled_circuits = {
                    k: qiskit.transpile(circuit, backend, optimization_level=0) for k, circuit in circuits.items()
                }

                # Run task.
                friend_size = f"{charlie_size}_{debbie_size}"
                friend_sizes.append(friend_size)
                print(f"Trial {trial} out of {num_trials} for task of {friend_size}")
                tasks[trial][friend_size] = backend.run(
                    list(transpiled_circuits.values()),
                    shots=shots,
                    verbatim=True,
                )
                print(f"Task with task ID: {tasks[trial][friend_size].job_id()}\n")

    results = {}
    post_processed_results: dict = {fs: {inequality: [] for inequality in ["semi_brukner"]} for fs in friend_sizes}
    for trial in tasks:
        for friend_size, task in tasks[trial].items():
            print(f"Processing trial {trial} for task with task ID: {task.job_id()}")

            result = task.result()
            for key, count in zip(transpiled_circuits.keys(), result.get_counts()):
                probabilities = {k[::-1]: v / shots for k, v in count.items()}
                results[key] = probabilities
            print(f"Results: {results}")

            charlie_size, debbie_size = map(int, friend_size.split("_"))

            # Compute violations from result counts.
            if strategy == "random":
                violations = compute_inequalities(results=results, verbose=verbose)
            elif strategy == "majority_vote":
                violations = compute_inequalities(
                    decode_results(results=results, charlie_size=charlie_size, debbie_size=debbie_size), verbose=verbose
                )
            print(f"Violations: {violations}\n")

            for key in violations:
                post_processed_results[friend_size][key].append(violations[key])
            print(f"Post-processed results: {post_processed_results}\n")

    return post_processed_results


def compute_inequalities(results: dict, verbose: bool = False) -> dict:
    """Compute the local friendliness inequalities from arXiv:1907.05607."""
    A1B2 = double_expect(("peek", "reverse_1"), results)
    A1B3 = double_expect(("peek", "reverse_2"), results)

    A3B2 = double_expect(("reverse_2", "reverse_1"), results)
    A3B3 = double_expect(("reverse_2", "reverse_2"), results)

    # Eq. (18) from [1].
    semi_brukner = -A1B2 + A1B3 - A3B2 - A3B3 - 2

    if verbose:
        print(f"{semi_brukner=} -- is violated: {semi_brukner > 0}")

    return {"semi_brukner": semi_brukner}


def single_expect(observer: int, setting: tuple[str, str], results: EWFSExpectations) -> float:
    """Compute single expectation values for either Alice or Bob."""
    if observer is ALICE:
        ret = 0
        for settings in results.keys():
            if settings[ALICE] is setting:
                probs = results[settings]
                # <A> = P(00) + P(01) - P(10) - P(11)
                ret += probs.get("00", 0) + probs.get("01", 0) - probs.get("10", 0) - probs.get("11", 0)
        return ret / len(SETTINGS)
    else:
        ret = 0
        for settings in results.keys():
            if settings[BOB] is setting:
                probs = results[settings]
                # <B> = P(00) - P(01) + P(10) - P(11)
                ret += probs.get("00", 0) - probs.get("01", 0) + probs.get("10", 0) - probs.get("11", 0)
        return ret / len(SETTINGS)


def double_expect(settings: tuple[str, str], results: EWFSExpectations) -> float:
    """Expectation value of product of two operators."""
    probs = results[settings]
    # <AB> = P(00) - P(01) - P(10) + P(11)
    return probs.get("00", 0) - probs.get("01", 0) - probs.get("10", 0) + probs.get("11", 0)


def decode_results(results: EWFSExpectations, charlie_size: int, debbie_size: int = 1) -> dict:
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


if __name__ == "__main__":
    settings = [
        ("peek", "reverse_1"),
        ("peek", "reverse_2"),
        ("reverse_2", "reverse_1"),
        ("reverse_2", "reverse_2"),
    ]
    compute_violations(settings=settings, strategy="random")
    compute_violations(settings=settings, strategy="majority_vote")
