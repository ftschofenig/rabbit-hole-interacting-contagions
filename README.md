# Conspiracy Theory Rabbit Holes Emerge via Interacting Contagions

This repository contains the public replication package for the paper
`Conspiracy Theory Rabbit Holes Emerge via Interacting Contagions`. It starts
from the included anonymized data archive and reproduces the analysis,
simulations, formal model results, main text figures, appendix figures, and
Table S1 calculations.

The committed PNG files in `figures_final` are the reference versions used by
the current manuscript. Running the notebooks regenerates them from the public
inputs.

## Companion repository

The embedding assisted narrative discovery and prompt tuning methodology is
documented in the
[companion repository](https://github.com/pvicinanza/llm_prompt_tuning_conspiracies).
The classifier output files and semantic distance matrix used by the present
paper are included in `anonymized_data.zip`, so reproducing the analyses does
not require running the companion repository.

## Scope

The public workflow contains eight notebooks and the package modules needed by
those notebooks. It does not contain the private source data preparation
workflow, cluster scheduling files, presentation exports, internal tests, or
cached intermediate results.

Figure S1 and Tables S2 and S3 concern the embedding and classifier methodology
documented in the companion repository. They are outside the executable
workflow of this repository. The present workflow uses the resulting classifier
output files and semantic distance matrix included in `anonymized_data.zip`.

## Repository contents

1. `anonymized_data.zip` contains the public inputs.

2. `conspiracy_analysis` contains the analysis and simulation package.

3. `config` contains the conspiracy name and inclusion configuration.

4. `notebooks` contains the complete public workflow.

5. `figures_final` contains the reference manuscript figures.

6. `requirements.txt` records the exact Python package versions used for the
   accepted reference run.

## Data extraction

Extract the archive from the repository root before running any notebook:

```bash
unzip anonymized_data.zip
```

The resulting layout must be:

```text
anonymized_data/
    G_MC.pkl
    full_tweet_0903.csv
    conspiracy_semantic_distance.csv
    labelled/
```

`full_tweet_0903.csv` contains only the column `Id`. It does not contain
usernames, tweet text, reply text, or retweet text. Researchers who require the
original tweet content must hydrate the retained Tweet IDs themselves, subject
to platform access, availability, and terms.

The sixteen files in `anonymized_data/labelled` contain only `tweet_id`,
`text_label`, and `yes_prob`. They contain no tweet text.

`G_MC.pkl` uses anonymous node identifiers and contains no usernames, tweet
text, profile fields, handles, or URLs. It necessarily retains network
topology, event timing, adoption histories, sharing histories, bot scores, and
exposure summaries required by the analyses. These structural data can carry
residual disclosure risk. Treat the archive as research data and do not attempt
to reidentify individuals.

## Environment

The accepted reference run used Python 3.11.3. Create an isolated environment
and install the exact dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Notebook execution order

Run the notebooks in the following order. The commands below write executed
copies to `executed_notebooks` and leave the committed source notebooks clean.
Each kernel runs with the source notebook directory as its working directory,
which is required by the relative paths in the notebooks.

First create the output directory:

```bash
mkdir -p executed_notebooks
```

Then run the empirical analysis:

```bash
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output-dir=executed_notebooks notebooks/01_analysis.ipynb
```

Run all three formal model notebooks. Their result bundles are independent, so
their order does not affect the combined manuscript figure:

```bash
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output-dir=executed_notebooks notebooks/formal_models/ecology_of_contagions_model.ipynb
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output-dir=executed_notebooks notebooks/formal_models/no_interaction_model.ipynb
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output-dir=executed_notebooks notebooks/formal_models/general_arousal_model.ipynb
```

The diagnostic files in `figures/formal_models` use shared filenames. They
therefore reflect the last formal model notebook run. The three separate result
bundles in `intermediate_files` are all retained and are combined by the main
text figure notebook.

Next run the full simulation:

```bash
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output-dir=executed_notebooks notebooks/02_simulation.ipynb
```

Generate the main text and appendix figures:

```bash
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output-dir=executed_notebooks notebooks/03_main_text_figures.ipynb
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output-dir=executed_notebooks notebooks/03_appendix_text_figures.ipynb
```

Finally run the threshold validation notebook for Table S1:

```bash
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output-dir=executed_notebooks notebooks/04_threshold_validation.ipynb
```

The Table S1 notebook displays the result table and prints its LaTeX form in
the executed notebook. It intentionally does not write a separate table file.

## Generated outputs

The workflow creates `intermediate_files` and writes figures to
`figures_final`, `figures_optional`, `figures/simulations`, and
`figures/formal_models`. Notebook 01 also writes three diagnostic PNG files
beside that notebook. Generated intermediates, logs, executed notebooks, and
nonmanuscript figure directories are ignored by Git.

The full reference workflow is computationally demanding. The accepted run
used an allocation with 128 logical CPUs and one terabyte of memory. That is an
allocation figure, not a measured peak memory value. The empirical analysis
took about eight hours and the simulation took about five hours. Runtime and
memory requirements will vary by hardware and process count.

## License

The software is released under the MIT License. See `LICENSE`.
