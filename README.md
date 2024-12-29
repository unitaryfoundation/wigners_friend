# Extended Wigner's Friend Scenario (EWFS)
Supplemental code for [arXiv:2409.15302](https://arxiv.org/abs/2409.15302) based on the experiment of
[arXiv:1907.05607](https://arxiv.org/abs/1907.05607)

## Installation
You will require **Python 3.12** and [`poetry`](https://python-poetry.org/). **Note:** the
newest Python version 3.13 is not yet supported due to the `qiskit-aer` dependency.

```sh
poetry install
```

## Usage
First, launch the virtual environment shell via poetry:

```sh
poetry shell
```

An EWFS circuit is comprised of two super-observers (Alice and Bob) who have respective friends (Charlie and Debbie). An
EWFS scenario consists of three settings that Alice and Bob can apply ("peek", "reverse_1", and "reverse_2").
Additionally, Alice and Bob can apply a specific strategy for an EWFS scenario ("random" or "majority_vote"). 

We can define an EWFS circuit of a specific size based on the quantum system sizes for Charlie and Debbie as well as the
setting choices and strategy for Alice and Bob.

```py
from ewfs.scenario import EWFS

# Example usage of the EWFS class.
ewfs = EWFS(
    alice_setting="peek",
    bob_setting="reverse_1",
    strategy="majority_vote",
    charlie_size=1,
    debbie_size=1,
)
circuit = ewfs.circuit()
print(circuit)
```

```sh
               ┌───┐┌───┐     ┌───────────┐┌───┐                                                       
Alice's qubit: ┤ X ├┤ H ├──■──┤ Rz(-2π/9) ├┤ H ├──■────────────────────────────────────────────────────
               ├───┤└───┘┌─┴─┐└─┬────────┬┘├───┤  │           ░      ┌───┐┌───────┐┌──────────┐┌───┐┌─┐
  Bob's qubit: ┤ X ├─────┤ X ├──┤ Rz(-π) ├─┤ H ├──┼────■──────░───■──┤ H ├┤ Rz(π) ├┤ Rz(π/18) ├┤ H ├┤M├
               └───┘     └───┘  └────────┘ └───┘┌─┴─┐  │  ┌─┐ ░   │  └───┘└───────┘└──────────┘└───┘└╥┘
      Charlie: ─────────────────────────────────┤ X ├──┼──┤M├─────┼──────────────────────────────────╫─
                                                └───┘┌─┴─┐└╥┘ ░ ┌─┴─┐                                ║ 
       Debbie: ──────────────────────────────────────┤ X ├─╫──░─┤ X ├────────────────────────────────╫─
                                                     └───┘ ║  ░ └───┘                                ║ 
measurement: 2/════════════════════════════════════════════╩═════════════════════════════════════════╩═
                                                           0                                         1 
```

We can compute the expectation value of a collection of circuits defined by setting choices over a chosen strategy by
running the circuits on either a real or simulated backend quantum device. Using these expectation values, we can
subsequently compute the Bell-like inequalities from [arXiv:1907.05607](https://arxiv.org/abs/1907.05607) (refer to
equations 13-21).

```py
from ewfs.experiment import run_experiment


settings = [
    ("peek", "reverse_1"),
    ("peek", "reverse_2"),
    ("reverse_2", "reverse_1"),
    ("reverse_2", "reverse_2"),
]
run_experiment(settings=settings, strategy="random")
```

Doing so will yield outputs of the resulting RHS of the inequalities for the choice of settings, strategy, and sizes for
Charlie and Debbie's systems.

```sh
Post-processed results: {'1_1': {'semi_brukner': [0.8317999999999999, 0.8263999999999996]}, '2_1': {'semi_brukner': [0.8126000000000002, 0.8027999999999995]}}
```

By default, the above runs on a simulator, however, we can also run on an IBM backend by supplying the `backend`
argument with an appropriate IBM Quantum backend object.

### Style guide
We don't have a style guide per se, but we recommend that both linter and formatter 
are run before each commit. In order to guarantee that, please install the pre-commit hook with

```sh
poetry run pre-commit install
``` 
immediately upon cloning the repository.

### Tests
The suite of unit tests can be run with
```sh
poetry run pytest
```

### Type checking
The project uses [mypy](https://mypy.readthedocs.io/en/stable/) for static type checking. To run mypy, use the following command:
```sh
poetry run mypy
```
