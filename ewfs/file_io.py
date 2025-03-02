"""Functionality for saving and loading experiment data."""

import os
import pickle
import qiskit
from ewfs.violations import compute_violations
from ewfs.setting import PEEK, REVERSE_1, REVERSE_2


def save_data(
    results: dict,
    charlie_size: int,
    debbie_size: int,
    trial: int,
    shots: int,
    backend: qiskit.providers.Backend,
    save_path: str | None = None,
) -> None:
    """Save EWFS file experiment datga.

    Writes data to a file name format of:
    `<MACHINE_NAME>_charlie_size_<CHARLIE_SIZE>_debbie_size_<DEBBIE_SIZE>_trial_<TRIAL>_shots_<NUM_SHOTS>`.
    """
    if save_path is None:
        # If not output file name is given, use this format.
        data_path = os.path.join(os.getcwd(), "data")
        try:
            backend_name = backend.name
        except AttributeError:
            backend_name = str(backend)
        output_file_name = (
            f"{backend_name}_charlie_size_{charlie_size}_debbie_size_{debbie_size}_trial_{trial}_shots_{shots}.pickle"
        )
        save_path = os.path.join(data_path, output_file_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    print(f"Writing data to: {save_path}")
    with open(save_path, "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_experiments(
    machine_name: str,
    friend_sizes: list[int],
    num_trials: int,
    shots: int,
    data_path: str,
    strategy: str,
) -> dict:
    """Load experiments from multiple files."""
    all_results: dict = {fs: {inequality: [] for inequality in ["semi_brukner"]} for fs in friend_sizes}

    for friend_size in friend_sizes:
        for trial in range(1, num_trials + 1):
            with open(
                os.path.join(data_path, f"{machine_name}_qubits_{friend_size}_trial_{trial}_shots_{shots}.pickle"), "rb"
            ) as file:
                results = pickle.load(file)
            mapping = {1: PEEK, 2: REVERSE_1, 3: REVERSE_2}
            results = {(mapping[k1], mapping[k2]): v for (k1, k2), v in results.items()}

            violations = compute_violations(results=results, charlie_size=friend_size, debbie_size=1, strategy=strategy)
            for key in violations:
                all_results[friend_size][key].append(violations[key])
    return all_results
