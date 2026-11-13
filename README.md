# TDT13 Project

This project investigates whether Reddit activity across subreddits can improve
prediction of users' self-reported Big Five personality scores, compared with
using comment text alone. It uses the PANDORA dataset and evaluates text,
subreddit, and combined models.

## Project structure

```text
src/
  preprocess_pandora.ipynb       Prepare the data and create cross-validation folds
  explore_pandora.ipynb           Explore the processed dataset
  T0_tfidf.ipynb                  TF-IDF text baseline
  T1_minilm.ipynb                 MiniLM text model
  T2_mpnet.ipynb                  MPNet text model
  S0_bag_of_subreddits.ipynb      Subreddit-count model
  S1_node2vec.ipynb               User-subreddit graph model
  C1_tfidf_subreddits.ipynb       Combined text and subreddit features
  C2_late_fusion.ipynb            Average text and subreddit predictions
  script.slurm                    Slurm job script for NTNU Idun
data/                             Local input data and generated results (not tracked)
```

## Data

The raw data and all generated datasets, embeddings, predictions, and metrics
are stored locally under `data/`. The entire directory is excluded by
`.gitignore`, so these files are **not included in this GitHub repository**.
You need to obtain the PANDORA data separately and place these files in `data/`:

- `author_profiles.csv`
- `all_comments_since_2015.csv`

The preprocessing notebook writes the processed data and fold assignments to
`data/preprocessed/`; the model notebooks write their outputs to subdirectories
under `data/`. The repository does not redistribute the dataset. Obtain and use
it in accordance with its source's access and usage terms.

## Models and evaluation

The notebooks compare:

- **Text:** TF-IDF over each user's comments, and averaged comment embeddings
  from the frozen MiniLM and MPNet sentence-transformer models.
- **Subreddit activity:** TF-IDF-transformed subreddit comment counts, and
  Node2Vec embeddings of the user-subreddit graph.
- **Combined models:** concatenated TF-IDF text and subreddit features, and
  late fusion by averaging text and subreddit test predictions.

The preprocessing notebook creates reproducible five-fold outer and inner
splits. The model notebooks tune Ridge regression on the inner folds and
evaluate on held-out outer folds for the five traits: agreeableness, openness,
conscientiousness, extraversion, and neuroticism. Reported metrics include
RMSE, Pearson correlation, and R-squared; baseline notebooks also compare
against predicting the training-fold mean.

## Running the notebooks

The notebooks expect to be run with `src/` as the working directory because
their data paths are relative to that directory. From the repository root:

```bash
cd src
jupyter lab
```

Run `preprocess_pandora.ipynb` first. Then run the desired model notebooks:

1. Run `T0_tfidf.ipynb`, `T1_minilm.ipynb`, and/or `T2_mpnet.ipynb` for text
   models.
2. Run `S0_bag_of_subreddits.ipynb` and/or `S1_node2vec.ipynb` for subreddit
   models.
3. Run `C1_tfidf_subreddits.ipynb` after the text TF-IDF baseline, or
   `C2_late_fusion.ipynb` after both the text and subreddit baselines.

`explore_pandora.ipynb` is optional. MiniLM and MPNet require
`sentence-transformers`; their first run downloads the selected pretrained
model from Hugging Face. Node2Vec requires the `node2vec` package. The
repository does not currently include a dependency lockfile, so set up a Python
environment with Jupyter, pandas, SciPy, scikit-learn, and the additional
packages required by the notebooks you plan to run.

## Running on NTNU Idun

The experiments are run on NTNU's Idun cluster as Slurm jobs. The checked-in
`src/script.slurm` is a starting point: it currently requests one CPU, 16 GB
memory, and up to 4 hours on the `CPUQ` partition, and activates the Conda
environment `tdt13idun`. Adapt the account, environment, resource requests, and
job output paths to your Idun setup.

