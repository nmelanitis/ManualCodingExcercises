# Exercise index

Open a starter below in VS Code and use **Run Cell** above each `# %%` section.
Replace each `...`, complete the code-only TODOs, then run the checks in that cell.
An unfinished task intentionally stops with an assertion or `NotImplementedError`.
Reset the kernel and run from the beginning when checking a finished worksheet.

## Suggested order

| Worksheet | Practice | Estimated time |
|---|---|---|
| [NumPy arrays](01_numpy/01_arrays_and_broadcasting.py) | Indexing, masks, axes, broadcasting, copies | 30–45 min |
| [pandas orders](02_pandas/01_clean_join_and_summarize.py) | Dirty types, duplicates, validated joins, groupby, pivots | 45–60 min |
| [Commute charts](03_visualization/01_commute_charts.py) | Distributions, group comparisons, scatterplots, figure export | 40–60 min |
| [Classification pipeline](04_sklearn/01_classification_pipeline.py) | Baselines, preprocessing, cross-validation, test metrics | 60–75 min |
| [OLS diagnostics](05_statsmodels/01_ols_and_diagnostics.py) | Formula models, confidence intervals, residuals, robust errors | 60–75 min |
| [PyTorch tensors/autograd](06_pytorch/01_tensors_and_autograd.py) | Shapes, forward passes, derivatives, gradient accumulation | 30–45 min |
| [PyTorch training loop](06_pytorch/02_classification_training_loop.py) | DataLoader, neural network, training/evaluation, model selection | 60–90 min |
| [Hugging Face tokenization](07_huggingface/01_tokenization_and_batches.py) | Dataset.map, vocabulary, padding, attention masks, model outputs | 60–75 min |
| [Pretrained classification](07_huggingface/02_pretrained_text_classification.py) | Auto classes, manual inference, pipeline comparison, truncation | 45–60 min |

The [general EDA template](../templates/eda/eda_template.py) is also available for
your own CSV datasets. Your existing `03_visualization/first_eda.py` is your attempt.

## How to practise

1. Select the project's `.venv` for both the Python interpreter and Jupyter kernel.
2. Use the ready-to-edit files here, or copy a fresh version from the matching
   topic under `templates/`. Use a new filename when repeating an exercise.
3. Type the core operations yourself. Imports and small datasets are supplied.
4. Check intermediate shapes, dtypes, and a few values before running assertions.
5. Finish with the written reflections and optionally attempt the stretch task.

All new worksheets except pretrained inference use in-memory or generated data,
with no dataset downloads. The offline Hugging Face model has random weights:
it teaches interfaces and batching, not meaningful sentiment predictions.

For pretrained inference, read the configuration cell: `ALLOW_DOWNLOAD=False`
uses only cached files. Set it to `True` for the first download if needed. That
worksheet runs inference locally and does not require a hosted inference service.

The training loop has mutable state. Re-running only its epoch cell continues
training the current model; re-run setup, preprocessing, and model creation for a
fresh experiment. A template's assertions check selected properties, not every
possible bug, so interpretation and a clean top-to-bottom run still matter.

Data Wrangler remains an optional inspection aid. The default workflow is manual
Python with Jupyter's Variables/Data Viewer, as described in the
[VS Code guide](../docs/vscode-guide.md).

Unused prepared imports are marked with `# ruff: noqa: F401` so Ruff's save actions do not
remove them before you reach their TODOs. Completed solutions should use separate
filenames; preserve your original attempts for review.

Useful help request: "Give me a hint for TODO 4 in the PyTorch training loop,
without writing the function for me."
