#!/usr/bin/env bash
#SBATCH --job-name=public_replication
#SBATCH --output=public_replication_%j.log
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=1024G

set -Eeuo pipefail

submit_pipeline() {
    local script_dir
    local script_path
    local run_tag
    local run_dir
    local analysis_ref
    local analysis_job
    local simulation_ref
    local simulation_job
    local figures_ref
    local figures_job

    script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
    script_path="$script_dir/run_full_pipeline.sh"
    cd "$script_dir"

    if [[ ! -f "requirements.txt" || ! -f "anonymized_data.zip" || ! -d "notebooks" ]]; then
        echo "This script must be located in the public repository root" >&2
        exit 2
    fi

    command -v sbatch >/dev/null || {
        echo "The sbatch command is unavailable" >&2
        exit 2
    }

    run_tag="$(date -u +%Y%m%dT%H%M%SZ)_$$"
    run_dir="$script_dir/logs/public_pipeline_$run_tag"
    mkdir -p "$run_dir" "$script_dir/executed_notebooks/$run_tag"
    printf 'notebook\tstatus\tseconds\tlog\n' > "$run_dir/status.tsv"
    date +%s > "$run_dir/run_started_epoch"
    touch "$run_dir/pipeline_started"

    analysis_ref=$(sbatch \
        --parsable \
        --job-name=public_analysis \
        --output="$run_dir/analysis_%j.log" \
        --export=ALL,PIPELINE_STAGE=analysis,PIPELINE_RUN_TAG="$run_tag" \
        "$script_path")
    analysis_job="${analysis_ref%%;*}"

    simulation_ref=$(sbatch \
        --parsable \
        --dependency="afterok:$analysis_job" \
        --kill-on-invalid-dep=yes \
        --job-name=public_simulation \
        --output="$run_dir/simulation_%j.log" \
        --export=ALL,PIPELINE_STAGE=simulation,PIPELINE_RUN_TAG="$run_tag" \
        "$script_path")
    simulation_job="${simulation_ref%%;*}"

    figures_ref=$(sbatch \
        --parsable \
        --dependency="afterok:$simulation_job" \
        --kill-on-invalid-dep=yes \
        --job-name=public_figures \
        --output="$run_dir/figures_%j.log" \
        --export=ALL,PIPELINE_STAGE=figures,PIPELINE_RUN_TAG="$run_tag" \
        "$script_path")
    figures_job="${figures_ref%%;*}"

    {
        printf 'run_tag\t%s\n' "$run_tag"
        printf 'analysis_job\t%s\n' "$analysis_job"
        printf 'simulation_job\t%s\n' "$simulation_job"
        printf 'figures_job\t%s\n' "$figures_job"
    } > "$run_dir/submitted_jobs.tsv"

    echo "Submitted the complete public replication pipeline"
    echo "Analysis job: $analysis_job"
    echo "Simulation job: $simulation_job"
    echo "Figure job: $figures_job"
    echo "Logs and status: $run_dir"
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    submit_pipeline
    exit 0
fi

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
PIPELINE_STAGE="${PIPELINE_STAGE:?PIPELINE_STAGE is not set}"
PIPELINE_RUN_TAG="${PIPELINE_RUN_TAG:?PIPELINE_RUN_TAG is not set}"
cd "$PROJECT_ROOT"

if [[ ! -f "requirements.txt" || ! -f "anonymized_data.zip" || ! -d "notebooks" ]]; then
    echo "The submission directory is not the public repository root: $PROJECT_ROOT" >&2
    exit 2
fi

EXECUTED_DIR="$PROJECT_ROOT/executed_notebooks/$PIPELINE_RUN_TAG"
NOTEBOOK_LOG_DIR="$PROJECT_ROOT/logs/public_pipeline_$PIPELINE_RUN_TAG"
STATUS_FILE="$NOTEBOOK_LOG_DIR/status.tsv"
RUN_MARKER="$NOTEBOOK_LOG_DIR/pipeline_started"
CURRENT_STEP="initial setup"

mkdir -p "$EXECUTED_DIR" "$NOTEBOOK_LOG_DIR"

report_unexpected_failure() {
    local status=$?
    echo "[$(date --iso-8601=seconds)] Pipeline failed during: $CURRENT_STEP" >&2
    echo "Status details: $STATUS_FILE" >&2
    exit "$status"
}

trap report_unexpected_failure ERR

fail() {
    echo "[$(date --iso-8601=seconds)] $*" >&2
    echo "Status details: $STATUS_FILE" >&2
    exit 1
}

run_notebook() {
    local notebook="$1"
    local filename="${notebook##*/}"
    local stem="${filename%.ipynb}"
    local log_file="$NOTEBOOK_LOG_DIR/${stem}.log"
    local started_at
    local finished_at
    local elapsed
    local status

    CURRENT_STEP="$notebook"
    started_at=$(date +%s)
    echo "[$(date --iso-8601=seconds)] Starting $notebook"

    if jupyter nbconvert \
        --to notebook \
        --execute \
        --ExecutePreprocessor.timeout=-1 \
        --output-dir="$EXECUTED_DIR" \
        "$notebook" 2>&1 | tee "$log_file"; then
        finished_at=$(date +%s)
        elapsed=$((finished_at - started_at))
        printf '%s\tpassed\t%s\t%s\n' "$notebook" "$elapsed" "$log_file" >> "$STATUS_FILE"
        echo "[$(date --iso-8601=seconds)] Finished $notebook in ${elapsed} seconds"
    else
        status=$?
        finished_at=$(date +%s)
        elapsed=$((finished_at - started_at))
        printf '%s\tfailed\t%s\t%s\n' "$notebook" "$elapsed" "$log_file" >> "$STATUS_FILE"
        echo "[$(date --iso-8601=seconds)] Failed $notebook with status $status" >&2
        return "$status"
    fi

    [[ -s "$EXECUTED_DIR/$filename" ]] || fail "Executed notebook was not saved: $EXECUTED_DIR/$filename"
}

load_python_module() {
    CURRENT_STEP="loading Python 3.11.3"
    module load python/3.11.3
}

activate_existing_environment() {
    [[ -x ".venv/bin/python" ]] || fail "The virtual environment is missing"
    source .venv/bin/activate
    local python_version
    python_version=$(python -c 'import platform; print(platform.python_version())')
    [[ "$python_version" == "3.11.3" ]] || fail "Expected Python 3.11.3 but found $python_version"
}

configure_runtime() {
    CURRENT_STEP="checking the Slurm allocation"
    local allocated_cpus="${SLURM_CPUS_PER_TASK:-0}"
    (( allocated_cpus >= 128 )) || fail "The notebooks use 128 workers but this job has $allocated_cpus CPUs"

    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export PYTHONUNBUFFERED=1
    export MPLBACKEND=Agg
}

prepare_fresh_inputs_and_environment() {
    CURRENT_STEP="creating the virtual environment"
    python3 -m venv --clear .venv
    source .venv/bin/activate

    CURRENT_STEP="installing Python dependencies"
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

    CURRENT_STEP="validating and extracting the public archive"
    command -v unzip >/dev/null || fail "The unzip command is unavailable"
    unzip -tq anonymized_data.zip
    unzip -qo anonymized_data.zip

    local required_inputs=(
        "anonymized_data/G_MC.pkl"
        "anonymized_data/full_tweet_0903.csv"
        "anonymized_data/conspiracy_semantic_distance.csv"
        "anonymized_data/labelled/labelled_5g.csv"
        "anonymized_data/labelled/labelled_antivax.csv"
        "anonymized_data/labelled/labelled_blm.csv"
        "anonymized_data/labelled/labelled_china.csv"
        "anonymized_data/labelled/labelled_deaths.csv"
        "anonymized_data/labelled/labelled_democrats.csv"
        "anonymized_data/labelled/labelled_fakenews.csv"
        "anonymized_data/labelled/labelled_fauci.csv"
        "anonymized_data/labelled/labelled_gates.csv"
        "anonymized_data/labelled/labelled_hospitals.csv"
        "anonymized_data/labelled/labelled_hydroxy.csv"
        "anonymized_data/labelled/labelled_pizzagate.csv"
        "anonymized_data/labelled/labelled_plandemic.csv"
        "anonymized_data/labelled/labelled_qanon.csv"
        "anonymized_data/labelled/labelled_testing.csv"
        "anonymized_data/labelled/labelled_trumppuppet.csv"
    )
    local input_path

    for input_path in "${required_inputs[@]}"; do
        [[ -s "$input_path" ]] || fail "Required public input is missing or empty: $input_path"
    done
}

validate_pipeline_outputs() {
    CURRENT_STEP="validating pipeline outputs"
    local required_outputs=(
        "intermediate_files/semantic_clustering.pkl"
        "intermediate_files/cox_bootstrap_results.pkl"
        "intermediate_files/timeline_bootstrap_results.pkl"
        "intermediate_files/ecology_of_contagions_model_results.pkl"
        "intermediate_files/no_interaction_model_results.pkl"
        "intermediate_files/general_arousal_model_results.pkl"
        "intermediate_files/simulation_results_all.pkl"
        "intermediate_files/simulation_results_all_manifest.json"
        "figures_final/fig01_formal_models.png"
        "figures_final/fig07_counterfactual_composite.png"
        "figures_final/fig08_combined_hazard_panel.png"
        "figures_final/fig09_settler_gateway_combined.png"
        "figures_final/figA01_per_conspiracy_prevalence.png"
        "figures_final/figA02_first_adoption_scatter.png"
        "figures_final/figA03_baseline_diffusion_curves.png"
        "figures_final/figA04_auc_total_all_scenarios.png"
        "figures_final/figA05_hawkes_goodness_of_fit.png"
        "figures_final/figA06_coadoption_jaccard.png"
        "figures_final/figA06_coadoption_jaccard_scatter.png"
        "figures_final/figA07_exposure_hr_comparison.png"
    )
    local output_path

    [[ -f "$RUN_MARKER" ]] || fail "The pipeline start marker is missing"

    for output_path in "${required_outputs[@]}"; do
        [[ -s "$output_path" ]] || fail "Required pipeline output is missing or empty: $output_path"
        [[ "$output_path" -nt "$RUN_MARKER" ]] || fail "Required pipeline output was not refreshed: $output_path"
    done
}

load_python_module

case "$PIPELINE_STAGE" in
    analysis)
        prepare_fresh_inputs_and_environment
        configure_runtime
        run_notebook "notebooks/01_analysis.ipynb"
        run_notebook "notebooks/formal_models/ecology_of_contagions_model.ipynb"
        run_notebook "notebooks/formal_models/no_interaction_model.ipynb"
        run_notebook "notebooks/formal_models/general_arousal_model.ipynb"
        run_notebook "notebooks/04_threshold_validation.ipynb"
        ;;
    simulation)
        activate_existing_environment
        configure_runtime
        run_notebook "notebooks/02_simulation.ipynb"
        ;;
    figures)
        activate_existing_environment
        configure_runtime
        run_notebook "notebooks/03_main_text_figures.ipynb"
        run_notebook "notebooks/03_appendix_text_figures.ipynb"
        validate_pipeline_outputs

        RUN_STARTED_EPOCH=$(<"$NOTEBOOK_LOG_DIR/run_started_epoch")
        PIPELINE_FINISHED_AT=$(date +%s)
        PIPELINE_ELAPSED=$((PIPELINE_FINISHED_AT - RUN_STARTED_EPOCH))
        printf 'passed\t%s\n' "$PIPELINE_ELAPSED" > "$NOTEBOOK_LOG_DIR/final_result.tsv"

        echo "[$(date --iso-8601=seconds)] Entire pipeline passed in ${PIPELINE_ELAPSED} seconds"
        echo "Executed notebooks: $EXECUTED_DIR"
        echo "Notebook logs: $NOTEBOOK_LOG_DIR"
        echo "Status details: $STATUS_FILE"
        ;;
    *)
        fail "Unknown pipeline stage: $PIPELINE_STAGE"
        ;;
esac

CURRENT_STEP="complete"
trap - ERR
deactivate
