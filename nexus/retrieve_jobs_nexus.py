import os
from datetime import datetime
import yaml
import logging

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

parser = argparse.ArgumentParser(prog='Retrieve job from Quantinuum hardware/emulator', description="")
parser.add_argument('--friend_size', type=int, help="Specify the friend size).", default=True)
parser.add_argument('--save', type=bool, help="Set to true if you want to save output files (default: True).", default=True)

args = parser.parse_args()

friend_size = args.friend_size
save = args.save

"""
Loads job names from a YAML file.
"""
PARAM_PATH = os.path.join(DATA_PATH, f"executed_jobs_names/friend_size_{friend_size}.yaml")
try:
    with open(PARAM_PATH, "r") as f:
        settings = yaml.safe_load(f)

    jobs = list(settings['job_names'])
    friend_size = int(settings['friend_size'])
    shots = int(settings['shots'])
    backend_name = str(settings['backend_name'])
    strategy = str(settings['strategy'])
    logging.info("Parameters loaded successfully.")
except Exception as e:
    logging.error(f"Error loading job names: {e}")


# Nexus setup:
project_name = "Wigners friend"
nexus_project = Nexus().get_project_by_name(project_name)
# config = AerConfig()
# backend_name = "simulator"

all_experiment_combos = [[PEEK, REVERSE_1], [PEEK, REVERSE_2], [REVERSE_2, REVERSE_1], [REVERSE_2, REVERSE_2]]

# Create timestamped directory to save results.
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
new_dir_name = f"{strategy}_{backend_name}_{timestamp}"
new_dir_path = os.path.join(DATA_PATH, new_dir_name)

if not os.path.exists(new_dir_path):
    os.makedirs(new_dir_path)

def retrieve_job(nexus_project, job_id, trial):
    results = {}
    
    execute_job_item = nexus_project.get_execute_job(name=job_id)
    result = execute_job_item.get_results()

    for i, res in enumerate(result):
        setting = all_experiment_combos[i]
        probabilities = {''.join(map(str, k)): v / shots for k, v in dict(res.get_counts()).items()}
        results[tuple(setting)] = probabilities

    violations = compute_violations(
        results=results, 
        charlie_size=friend_size, 
        debbie_size=1, 
        strategy=strategy,
        verbose=True,
    )
    print(f"Violations: {violations}\n")

    if save:
        save_data(results=results,
                  friend_size=friend_size,
                  trial=1+trial,
                  shots=shots,
                  data_path=new_dir_path,
                  backend_name=backend_name)

print(jobs, friend_size, shots, backend_name, strategy)
    
for trial, job in enumerate(jobs):
    retrieve_job(nexus_project, job, trial)


