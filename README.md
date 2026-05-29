# Stats 292 — Statistical Models of Text and Language

**Stanford University, Spring 2026**
**Instructor:** Prof. David Donoho

## Homework Assignments

| File | Format | Topic | Due |
|---|---|---|---|
| `HW_FreqStats_BSky.ipynb` | Jupyter Notebook | Frequency statistics on Bluesky skeets | April 16, 2026 at 23:59 PDT |
| `HW_FreqStats_BSky.md` | Markdown | Same assignment (reference/print version) | April 16, 2026 at 23:59 PDT |
| `HW_POSTagger_BSky.ipynb` | Jupyter Notebook | Part-of-speech tagging via Hidden Markov Models | April 28, 2026 at 23:59 PDT |
| `HW_POSTagger_BSky.md` | Markdown | Same assignment (reference/print version) | April 28, 2026 at 23:59 PDT |
| `HW3_CountryCapital_WordGeometry.Rmd` | R Markdown | Word vector geometry: countries, capitals, and development | May 7, 2026 at 23:59 PDT |
| `HW3_CountryCapital_WordGeometry.ipynb` | Jupyter Notebook (R kernel) | Same assignment — use if you prefer Jupyter over RStudio | May 7, 2026 at 23:59 PDT |
| `HW4_ResidualStream_LogitLens.ipynb` | Jupyter Notebook (Google Colab) | Residual stream: logit lens, mass-mean probe, directional ablation | May 21, 2026 at 23:59 PDT |
| `HW5_AttentionCircuits.ipynb` | Jupyter Notebook | Attention circuits: OV/QK virtual weights, induction heads, causal ablation, composition scores | June 5, 2026 at 23:59 PDT |

## Opening HW4 in Google Colab

HW4 runs on **Google Colab** (free tier is sufficient for Parts 1–4):

1. Download `HW4_ResidualStream_LogitLens.ipynb` from this repo.
2. Go to [colab.research.google.com](https://colab.research.google.com) → **File → Open notebook → Upload tab**.
3. Drag the `.ipynb` file into the upload dialog, or click **Browse** to select it.
4. Run the Setup cells at the top in order — they install `transformer_lens`, download GPT-2 weights (~500 MB, ~1 min), and clone the Marks & Tegmark dataset. **Do not use "Run all" until setup is complete.**
5. Part 5.4 (GPT-2 medium, optional extension) benefits from a GPU: **Runtime → Change runtime type → T4 GPU**.

## Running HW5

HW5 runs locally using the `stats292` conda environment (no Colab required):

1. Update your environment from `environment.yml` (adds `transformer_lens`, `einops`, `datasets`):
   ```bash
   conda env update -f environment.yml --prune
   conda activate stats292
   ```
2. Open `HW5_AttentionCircuits.ipynb` in JupyterLab and select the **Python (stats292)** kernel.
3. Run the setup cell — it loads GPT-2 small (~500 MB, ~30 sec on first run; cached after that).
4. Parts 1–4 run on CPU. **No GPU required.**

## Getting Started

### 1. Install the Conda environment

```bash
conda env create -f environment.yml
conda activate stats292
```

### 2. Download NLTK data

```bash
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')"
```

### 3. Authenticate with Google Cloud

```bash
gcloud auth application-default login
```

Sign in with your Stanford Google account when the browser opens.

### 4. Register the Jupyter kernel

```bash
python -m ipykernel install --user --name stats292 --display-name "Python (stats292)"
```

### 5. Open the notebook

Open `HW_FreqStats_BSky.ipynb` in JupyterLab or VS Code, then select the **Python (stats292)** kernel.

## Updating the environment

If `environment.yml` is updated during the quarter:

```bash
conda env update -f environment.yml --prune
```
