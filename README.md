# Sticks and Carrots Affect LLMs’ Self-Confidence - On the Connection Between Safety and Sentiment Features
## Introduction
M.Sc. project in the course "Interpretability of Large Language Models" at Tel Aviv University. We use Gemma Scope SAEs to steer Gemma model
activations along sentiment and jailbreak directions, then measure how this affects refusal rates and response quality.
You can clone the repository and change the code to fit any other model.

## What's in here
- `refusal_results.py` — Tkinter UI for labeling steered responses (refusal yes/no, quality 1-10), plus code to aggregate
  results and generate figures
- `final_project_notebook.ipynb` — Notebook with the SAE analysis and steering experiments
- `steering results/` — Raw JSON outputs from steering experiments, organized by model (12b, 4b) and steering type (
  positive, negative, jailbreak, and combinations)
- `labels.json` — Human annotations from the labeling UI
- `final_analysis.csv`  — Aggregated metrics per steering configuration
- `figures/` — Generated plots (PDF + PNG)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas matplotlib numpy
```

## Usage

Run the labeling UI and generate figures:

```bash
python3 refusal_results.py
```

The script opens a Tkinter window where you label each steered response. Progress saves to `labels.json` automatically.
When you close the window, it aggregates results to CSV and generates figures.

## Steering dimensions

- **Positive/Negative sentiment** — steering along sentiment SAE features with varying coefficients
- **Jailbreak** — steering along jailbreak-related features
- **Combined** — applying both sentiment and jailbreak steering simultaneously

### Contact
For questions or suggestions, please contact Alon Kaplan (alonkaplan1@mail.tau.ac.il) Emile Mosseri (emile.mosseri@mail.tau.ac.il) 