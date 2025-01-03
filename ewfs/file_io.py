"""Functionality for saving and loading experiment data."""

import os
import pickle
import qiskit


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
