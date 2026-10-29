# Manual Coding Exercises

A practice-first workspace for writing data-science code by hand in VS Code.
Templates use `# %%` cells, so they behave like a Python script and a notebook at
the same time.

## Project layout

```text
.
├── .vscode/                 # Workspace settings and extension suggestions
├── data/
│   ├── raw/                 # Original data; do not modify in place
│   ├── interim/             # Temporary cleaning results
│   ├── processed/           # Analysis-ready data
│   └── sample/              # Small, shareable practice datasets
├── docs/                    # Workflow guides
├── exercises/
│   ├── 01_numpy/
│   ├── 02_pandas/
│   ├── 03_visualization/
│   ├── 04_sklearn/
│   ├── 05_statsmodels/
│   ├── 06_pytorch/
│   └── 07_huggingface/
├── outputs/
│   ├── figures/
│   └── tables/
├── templates/
│   ├── eda/                 # General EDA worksheet
│   ├── numpy/               # Arrays and broadcasting
│   ├── pandas/              # Cleaning, joins, and summaries
│   ├── visualization/       # seaborn and Matplotlib
│   ├── sklearn/             # Classification pipeline
│   ├── statsmodels/         # OLS and diagnostics
│   ├── pytorch/             # Tensors/autograd and a training loop
│   └── huggingface/         # Tokenization and pretrained inference
├── tests/
├── AGENTS.md                # Makes the project rules visible to Codex
└── skills.md                # Learning and collaboration contract
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Then open the folder in VS Code, run **Python: Select Interpreter**, and choose
`.venv`. See [the VS Code guide](docs/vscode-guide.md) for the recommended daily
workflow and shortcuts.

## Start the first exercise

1. Copy `templates/eda/eda_template.py` to
   `exercises/03_visualization/my_first_eda.py`.
2. Put a CSV in `data/raw/` and update `DATA_PATH` in your copy.
3. Complete one numbered `TODO` at a time with `Shift+Enter`.
4. Write observations as you go; do not leave interpretation until the end.

## Continue practising

The [exercise index](exercises/README.md) lists the worksheets in suggested order.
Ready-to-edit starter copies are already in the topic folders under `exercises/`;
use `templates/` whenever you want a fresh attempt.

For deep learning, start with PyTorch tensors/autograd, continue to the training
loop, then work through Hugging Face tokenization and pretrained inference.
The first three run on CPU with locally supplied/generated data. The pretrained
worksheet uses a model download or an existing local cache, as explained in its
configuration cell.
