#!/usr/bin/env bash
#SBATCH --job-name=public_replication
#SBATCH --output=public_replication_%j.txt
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=1024G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"

EXECUTED_DIR="$SLURM_SUBMIT_DIR/executed_notebooks"
mkdir -p "$EXECUTED_DIR"

echo "[$(date)] Creating the Python environment"
module load python/3.11.3
python3 -m venv --clear .venv
source .venv/bin/activate
python --version
python -m pip install -r requirements.txt

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

echo "[$(date)] Extracting anonymized_data.zip"
unzip -qo anonymized_data.zip

cd notebooks

echo "[$(date)] Starting 01_analysis.ipynb"
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output-dir="$EXECUTED_DIR" 01_analysis.ipynb
echo "[$(date)] Finished 01_analysis.ipynb"

cd formal_models

echo "[$(date)] Starting ecology_of_contagions_model.ipynb"
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output-dir="$EXECUTED_DIR" ecology_of_contagions_model.ipynb
echo "[$(date)] Finished ecology_of_contagions_model.ipynb"

echo "[$(date)] Starting no_interaction_model.ipynb"
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output-dir="$EXECUTED_DIR" no_interaction_model.ipynb
echo "[$(date)] Finished no_interaction_model.ipynb"

echo "[$(date)] Starting general_arousal_model.ipynb"
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output-dir="$EXECUTED_DIR" general_arousal_model.ipynb
echo "[$(date)] Finished general_arousal_model.ipynb"

cd ..

echo "[$(date)] Starting 04_threshold_validation.ipynb"
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output-dir="$EXECUTED_DIR" 04_threshold_validation.ipynb
echo "[$(date)] Finished 04_threshold_validation.ipynb"

echo "[$(date)] Starting 02_simulation.ipynb"
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output-dir="$EXECUTED_DIR" 02_simulation.ipynb
echo "[$(date)] Finished 02_simulation.ipynb"

echo "[$(date)] Starting 03_main_text_figures.ipynb"
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output-dir="$EXECUTED_DIR" 03_main_text_figures.ipynb
echo "[$(date)] Finished 03_main_text_figures.ipynb"

echo "[$(date)] Starting 03_appendix_text_figures.ipynb"
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output-dir="$EXECUTED_DIR" 03_appendix_text_figures.ipynb
echo "[$(date)] Finished 03_appendix_text_figures.ipynb"

echo "[$(date)] Entire pipeline finished"
