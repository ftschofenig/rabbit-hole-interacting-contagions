# Conspiracy Theory Rabbit Holes Emerge via Interacting Contagions

This repository contains the public replication package for *Conspiracy Theory
Rabbit Holes Emerge via Interacting Contagions*. It includes the anonymized
inputs, analysis code, simulations, and notebooks needed to reproduce the
reported results and figures.

## Companion project

The embedding assisted narrative discovery and prompt tuning methodology is
available in the [companion repository](https://github.com/pvicinanza/llm_prompt_tuning_conspiracies).
The resulting classifier outputs and semantic distance matrix are already
included in this repository's data archive.

## Project structure

1. `anonymized_data.zip` contains the public data required by the notebooks.

2. `conspiracy_analysis` contains the analysis, modelling, simulation, and
   visualization code.

3. `config` contains the conspiracy category configuration.

4. `notebooks` contains the executable analysis pipeline.

5. `figures_final` contains the manuscript reference figures.

6. `requirements.txt` lists the Python dependencies.

7. `run_full_pipeline.sh` runs a complete validation of the public workflow on
   a Slurm cluster.

## Setup

Use Python 3.11.3 and install the dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

From the repository root, unpack the data archive:

```bash
unzip anonymized_data.zip
```

The extracted `anonymized_data` directory must remain in the repository root,
beside `conspiracy_analysis`, `config`, and `notebooks`:

```text
repository/
    anonymized_data/
    conspiracy_analysis/
    config/
    notebooks/
```

## Running the pipeline

On a Slurm cluster, submit the complete workflow from the repository root:

```bash
./run_full_pipeline.sh
```

The script submits dependent jobs for the analysis, simulation, and figure
stages. It creates the Python environment, installs the dependencies, validates
and extracts the archive, stops after any failed notebook, and records executed
notebooks and logs under `executed_notebooks` and `logs`.

For manual execution, run the notebooks in this order. Each notebook must use
its containing directory as the kernel working directory.

1. `notebooks/01_analysis.ipynb`

2. The three notebooks in `notebooks/formal_models/`

3. `notebooks/04_threshold_validation.ipynb`

4. `notebooks/02_simulation.ipynb`

5. `notebooks/03_main_text_figures.ipynb`

6. `notebooks/03_appendix_text_figures.ipynb`

The analysis notebook creates the empirical inputs used by the simulation and
figure stages.
The simulation notebook creates the simulation results used by the figure
notebooks. The formal model result bundles can be created in any order.

## Notebook outputs

1. `01_analysis.ipynb` fits the empirical models and creates the intermediate
   analysis results used by the simulation and figure notebooks.

2. The notebooks in `formal_models/` create the three formal model result sets
   used in the main text figures.

3. `04_threshold_validation.ipynb` displays the Table S1 classifier threshold
   calculation in the executed notebook.

4. `02_simulation.ipynb` creates the simulation results and simulation
   diagnostics.

5. `03_main_text_figures.ipynb` creates the main text figures.

6. `03_appendix_text_figures.ipynb` creates the appendix figures.

## Computing requirements

The full pipeline is computationally demanding and is intended for an HPC
environment. The automated runner requests three dependent jobs with up to 48
hours, 128 CPUs, and one terabyte of memory for each job. The analysis and
simulation notebooks both use 128 worker processes. Reduce those settings
before using a smaller allocation.

## License

The software is released under the MIT License. See `LICENSE`.
