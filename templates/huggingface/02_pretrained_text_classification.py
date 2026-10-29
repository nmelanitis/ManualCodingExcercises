"""Hugging Face: inspect pretrained sentiment predictions. 45–60 minutes.

This worksheet uses an English sentiment checkpoint. Its first uncached load
downloads model/tokenizer files when ALLOW_DOWNLOAD=True. Inference runs locally
on CPU. If you prefer no downloads, use 01_tokenization_and_batches.py.
"""
# ruff: noqa: F401

# %% [markdown]
# # Pretrained text classification
#
# **Goal:** understand the operations hidden by a sentiment-analysis pipeline
# and inspect its behavior on clear, negated, and ambiguous sentences.

# %% 0. Configuration and fictional inputs
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
MODEL_REVISION = "main"  # Optional: replace with a commit hash for reproducibility.
ALLOW_DOWNLOAD = False  # Set True yourself for the first uncached load.
MAX_LENGTH = 128
DEVICE = torch.device("cpu")
texts = [
    "The film was wonderful and I would watch it again.",
    "The film was boring and the dialogue was terrible.",
    "I expected to hate it, but I really enjoyed it.",
    "The acting was great, but the story was disappointing.",
    "It was not bad.",
    "This film exists.",
]
start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PROJECT_ROOT = next(
    (p for p in [start, *start.parents] if (p / "skills.md").is_file()), None
)
if PROJECT_ROOT is None:
    raise RuntimeError("Run from inside the project.")
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# %% 1. Load a matching tokenizer and model
# TODO 1: Use AutoTokenizer.from_pretrained and
# AutoModelForSequenceClassification.from_pretrained with MODEL_ID.
# Pass revision=MODEL_REVISION and local_files_only=not ALLOW_DOWNLOAD to BOTH.
# Use the same checkpoint for both, move the model to DEVICE, and call eval().
# If the cache is empty and downloads are disabled, loading will fail: either
# enable the explicit download switch or work through the offline worksheet.
tokenizer = ...
model = ...
assert tokenizer is not Ellipsis and model is not Ellipsis, "Complete TODO 1."
assert model.config.num_labels == 2 and not model.training
print("Label mapping:", model.config.id2label)

# %% 2. Tokenize a batch
# TODO 2: Tokenize texts with padding=True, truncation=True,
# max_length=MAX_LENGTH, return_tensors="pt". Move inputs to DEVICE.
inputs = ...
assert inputs is not Ellipsis, "Complete TODO 2."
assert inputs["input_ids"].shape[0] == len(texts)
assert inputs["attention_mask"].shape == inputs["input_ids"].shape
assert inputs["input_ids"].shape[1] <= MAX_LENGTH
# Inspect tokens for "It was not bad." and identify any padding positions.

# %% 3. Run inference manually
with torch.inference_mode():
    # TODO 3: Call model(**inputs) and retain the output object.
    outputs = ...
assert outputs is not Ellipsis, "Complete TODO 3."
assert outputs.logits.shape == (len(texts), 2)
assert not outputs.logits.requires_grad

# %% 4. Decode predictions using the checkpoint metadata
# TODO 4: Apply softmax to logits across classes, select the highest-probability
# class ids, and map those ids through model.config.id2label.
# Extract each selected class's probability as its score.
probabilities = ...
predicted_ids = ...
predicted_labels = ...
predicted_scores = ...
assert probabilities is not Ellipsis and predicted_labels is not Ellipsis
torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(len(texts)))
assert len(predicted_labels) == len(predicted_scores) == len(texts)
assert set(predicted_labels) <= set(model.config.id2label.values())

# %% 5. Build an inspection table
# TODO 5: Create a DataFrame with text, label, and score columns.
# Preserve input order, then inspect the most uncertain examples separately.
prediction_table = ...
assert isinstance(prediction_table, pd.DataFrame), "Complete TODO 5."
assert prediction_table["text"].tolist() == texts
assert prediction_table["score"].between(0, 1).all()
print(prediction_table)
# Optional Data Wrangler: inspect this table alongside the usual Data Viewer.

# %% 6. Compare with the high-level pipeline
# TODO 6: Create pipeline("text-classification", model=model,
# tokenizer=tokenizer, device=-1) using the objects already loaded.
# Call it on texts using batch_size=3, truncation=True, max_length=MAX_LENGTH.
sentiment_pipeline = ...
pipeline_results = ...
assert pipeline_results is not Ellipsis, "Complete TODO 6."
assert len(pipeline_results) == len(texts)
assert [item["label"] for item in pipeline_results] == list(predicted_labels)
torch.testing.assert_close(
    torch.tensor([item["score"] for item in pipeline_results]),
    torch.tensor(list(predicted_scores), dtype=torch.float32),
    atol=1e-5,
    rtol=1e-4,
)
# Explain how tokenization, batching, inference, and decoding map to earlier cells.

# %% 7. Investigate truncation
long_text = "The opening scene was ordinary. " * 60 + "I hated the ending."
# TODO 7: Tokenize long_text once without truncation and once with MAX_LENGTH.
# Compare lengths, then decode the truncated input. Was the ending retained?
full_encoding = ...
short_encoding = ...
assert full_encoding is not Ellipsis and short_encoding is not Ellipsis
assert len(full_encoding["input_ids"]) > MAX_LENGTH
assert len(short_encoding["input_ids"]) == MAX_LENGTH
# Write your decoded-text inspection here.

# %% 8. Save results
# TODO 8: Save prediction_table to TABLE_DIR / "pretrained_sentiment.csv",
# with index=False. This exports predictions, not a downloaded model.
# Write the export call here.
assert (TABLE_DIR / "pretrained_sentiment.csv").is_file(), "Complete TODO 8."

# %% [markdown]
# ## TODO 9: Evaluate critically
#
# - Which examples were clear or ambiguous to you? Compare with the predictions.
# - Does the model have a neutral class? What happens to a neutral sentence?
# - Why is the softmax score not a guarantee of correctness or calibration?
# - How could truncation hide the part of a review carrying its sentiment?
# - Read the checkpoint's model card: what data, language, and limitations matter?
#
# Optional stretch: label 20 new examples yourself before prediction, then compare
# accuracy with a majority-class baseline. Include negation and mixed sentiment.
# Keep this an inference exercise; use a separate train/validation/test setup for
# future fine-tuning.
#
# VS Code practice: use the Variables/Data Viewer for prediction_table, and hover
# over from_pretrained to inspect local_files_only and revision options.
# Model card:
# https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english
