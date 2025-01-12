"""Run the EWFS experiment."""

from collections import defaultdict
import qiskit
import qiskit_aer
from qiskit_ibm_runtime import SamplerV2

from ewfs.file_io import save_data
from ewfs.scenario import EWFS
from ewfs.violations import compute_violations
from ewfs.setting import SETTING_PAIRS


def run_experiment(
    shots: int = 10_000,
    num_trials: int = 2,
    charlie_sizes: range | None = None,
    debbie_sizes: range | None = None,
    strategy: str = "majority_vote",
    behavior: str = "cnot_ladder",
    backend: qiskit.providers.Backend | None = None,
    settings: list[tuple[str, ...]] | None = None,
    save: bool = False,
    save_path: str | None = None,
) -> dict:
    """Run the EWFS experiment."""
    settings = settings or SETTING_PAIRS
    backend = backend or qiskit_aer.Aer.get_backend("aer_simulator")
    sampler = SamplerV2(backend)

    charlie_sizes = charlie_sizes or range(1, 3)
    debbie_sizes = debbie_sizes or range(1, 2)

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
                for alice_setting, bob_setting in settings:
                    circuit = EWFS(
                        alice_setting=alice_setting,
                        bob_setting=bob_setting,
                        behavior=behavior,
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
                tasks[trial][friend_size] = sampler.run(
                    list(transpiled_circuits.values()),
                    shots=shots,
                )
                print(f"Task with task ID: {tasks[trial][friend_size].job_id()}\n")

    results = {setting: {"00": 0.0, "01": 0.0, "10": 0.0, "11": 0.0} for setting in settings}
    post_processed_results: defaultdict = defaultdict(lambda: defaultdict(list))
    for trial in tasks:
        for friend_size, task in tasks[trial].items():
            print(f"Processing trial {trial} for task with task ID: {task.job_id()}")

            result = task.result()
            pub_results = zip(
                transpiled_circuits.keys(), [pub_result.data.measurement.get_counts() for pub_result in result]
            )
            for key, count in pub_results:
                probabilities = {k[::-1]: v / shots for k, v in count.items()}
                results[key] = probabilities
            print(f"Results: {results}")

            # friend_size is a string of the form "charlie_size_debbie_size".
            charlie_size, debbie_size = map(int, friend_size.split("_"))

            # Compute violations from result counts.
            violations = compute_violations(
                results=results,
                charlie_size=charlie_size,
                debbie_size=debbie_size,
                strategy=strategy,
                verbose=True,
            )
            print(f"Violations: {violations}\n")

            for key in violations:
                post_processed_results[friend_size][key].append(violations[key])
            print(f"Post-processed results: {post_processed_results}\n")

    if save:
        save_data(
            results=results,
            charlie_size=charlie_size,
            debbie_size=debbie_size,
            trial=trial,
            shots=shots,
            backend=backend,
            save_path=save_path,
        )
    return post_processed_results
