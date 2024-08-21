import os
from datetime import datetime


from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit.visualization import plot_distribution, plot_histogram

from pytket.extensions.nexus import Nexus

from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from pytket.extensions.nexus import Nexus, QuantinuumConfig
from pytket.extensions.nexus.backends import NexusBackend

from utils import depolarizing_noise_model, bitflip_model, DATA_PATH, save_data

from ewfs import ewfs, PEEK, REVERSE_1, REVERSE_2, ANGLES, BETA, decode_results, compute_violations


import argparse

parser = argparse.ArgumentParser(prog='Run EWFS on Quantinuum hardware/emulator', description="")
parser.add_argument('--backend', type=str, help="Specify backend.")
parser.add_argument('--friend_size', type=int, help="Set the size of Charlie.")
parser.add_argument('--shots', type=int, help="Set number of shots.")
parser.add_argument('--trials', type=int, help="Set number of trials.")
parser.add_argument('--save', type=bool, help="Set to true if you want to save output files (default: True).", default=True)

args = parser.parse_args()

# Experimental settings:
shots = args.shots
num_trials = args.trials
friend_sizes = [args.friend_size]
strategy = "majority_vote"
save = args.save

# Nexus setup:
project_name = "Wigners friend"
nexus_project = Nexus().get_project_by_name(project_name)
# config = AerConfig()
# backend_name = "simulator"

# Create timestamped directory to save results.
backend_name = args.backend
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
new_dir_name = f"{strategy}_{backend_name}_{timestamp}"
new_dir_path = os.path.join(DATA_PATH, new_dir_name)

if not os.path.exists(new_dir_path):
    os.makedirs(new_dir_path)

# Example of building a NexusBackend from a QuantinuumConfig
configuration = QuantinuumConfig(device_name=backend_name, user_group="DEFAULT")
backend = NexusBackend(configuration, nexus_project)

# backend.cost(circuit, shots)

all_experiment_combos = [[PEEK, REVERSE_1], [PEEK, REVERSE_2], [REVERSE_2, REVERSE_1], [REVERSE_2, REVERSE_2]]

def run_experiment(charlie_size, trial, all_experiment_combos, configuration, nexus_project, backend, shots, strategy, save, new_dir_path, backend_name):
    results = {}
    qcs = []
    for setting in all_experiment_combos:
        # Create timestamped directory to save results.
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        qc = qiskit_to_tk(ewfs(setting[0], setting[1], "majority_vote", ANGLES, BETA, charlie_size, 1))
        qcs.append(qc)
        
    nexus_compile_job = nexus_project.submit_compile_job(
        backend_config=configuration,
        circuits=qcs,
        name=f"compile_job_charlie_size_{charlie_size}_{timestamp}",
        optimisation_level=0,
    )
    print("sending for compile", setting)
    nexus_compile_job.wait_for()
    print("got compile")
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    execute_job_name = f"execute_job_charlie_size_{charlie_size}_{timestamp}"

    compiled_circuits = nexus_compile_job.get_compiled_circuits()
    print("asking for cost")

    cost = backend.cost(compiled_circuits[0], shots, 'H1-1SC')
    print("Estimated Cost: ", cost)
    
    nexus_execute_job = nexus_project.submit_execute_job(
        backend_config=configuration,
        circuits=compiled_circuits,
        n_shots=[shots]*len(compiled_circuits),
        name=execute_job_name,
    )

    nexus_execute_job.wait_for()

    # wait days maybe
    result = nexus_execute_job.get_results()

    for i, res in enumerate(result):
        setting = all_experiment_combos[i]
        probabilities = {''.join(map(str, k)): v / shots for k, v in dict(res.get_counts()).items()}
        results[tuple(setting)] = probabilities

    violations = compute_violations(
        results=results, 
        charlie_size=charlie_size, 
        debbie_size=1, 
        strategy=strategy,
        verbose=True,
    )
    print(f"Violations: {violations}\n")

    if save:
        save_data(results=results,
                  friend_size=charlie_size,
                  trial=trial,
                  shots=shots,
                  data_path=new_dir_path,
                  backend_name=backend_name)

for charlie_size in friend_sizes:
    for trial in range(num_trials):
        run_experiment(charlie_size, trial, all_experiment_combos, configuration, nexus_project, backend, shots, strategy, save, new_dir_path, backend_name)

