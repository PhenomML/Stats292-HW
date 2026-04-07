# Stats 292 — Statistical Models of Text and Language

**Stanford University, Spring 2026**
**Instructor:** Prof. David Donoho

## Homework Assignments

| File | Format | Topic | Due |
|---|---|---|---|
| `HW_FreqStats_BSky.ipynb` | Jupyter Notebook | Frequency statistics on Bluesky skeets | April 16, 2026 at 23:59 PDT |
| `HW_FreqStats_BSky.md` | Markdown | Same assignment (reference/print version) | April 16, 2026 at 23:59 PDT |

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
