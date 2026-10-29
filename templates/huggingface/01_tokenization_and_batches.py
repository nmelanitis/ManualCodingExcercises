"""Hugging Face: datasets, tokenization, padding, and model inputs. 60–75 minutes.

Everything runs locally using a tiny vocabulary and a randomly initialized BERT.
The random model is for learning tensor interfaces; its predictions have no
trained sentiment meaning. Complete TODOs in order using VS Code Run Cell.
"""
# ruff: noqa: F401

# %% [markdown]
# # From text to a model batch
#
# **Goal:** inspect each representation between raw text and model output.

# %% 0. Supplied fictional examples and imports
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordLevelTrainer
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    DataCollatorWithPadding,
    PreTrainedTokenizerFast,
)

RANDOM_SEED = 42
MAX_LENGTH = 32
BATCH_SIZE = 4
torch.manual_seed(RANDOM_SEED)
texts = [
    "A lovely film",
    "A dull film",
    "I enjoyed the story",
    "I disliked the story",
    "The acting was excellent",
    "The acting was poor",
    "Wonderful music and a warm ending",
    "Loud music and a boring ending",
    "A funny and thoughtful comedy",
    "An unfunny and confusing comedy",
    "Beautiful scenes",
    "Ugly scenes",
    "The characters felt real",
    "The characters felt flat",
    "A delightful short adventure",
    "A disappointing long adventure",
    "I would watch this again",
    "I would never watch this again",
    "A charming surprise",
    "A tedious surprise",
]
labels = [1, 0] * 10

# %% 1. Create and split a Dataset
# TODO 1: Create Dataset.from_dict with columns text and labels.
# Split with test_size=0.2 and seed=RANDOM_SEED.
# This tiny split teaches the API; it is not a useful performance benchmark.
dataset = ...
splits = ...
assert isinstance(dataset, Dataset), "Complete TODO 1."
assert len(splits["train"]) == 16 and len(splits["test"]) == 4
assert set(splits["train"]["text"]).isdisjoint(splits["test"]["text"])

# %% 2. Train a tiny tokenizer on training text only
# This setup supplies normalization and pre-tokenization plumbing.
backend = Tokenizer(WordLevel(unk_token="[UNK]"))
backend.normalizer = Lowercase()
backend.pre_tokenizer = Whitespace()
trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]"])
# TODO 2a: Train backend on splits["train"]["text"] using train_from_iterator,
# passing trainer=trainer. Do not include the held-out text.
# Write the training call here.
assert backend.get_vocab_size() > 4, "Complete TODO 2a."

# Supply the special-token layout expected by a BERT-like input.
backend.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    special_tokens=[
        ("[CLS]", backend.token_to_id("[CLS]")),
        ("[SEP]", backend.token_to_id("[SEP]")),
    ],
)
# TODO 2b: Wrap backend in PreTrainedTokenizerFast(tokenizer_object=...).
# Set unk_token, pad_token, cls_token, sep_token, and model_max_length=MAX_LENGTH.
tokenizer = ...
assert isinstance(tokenizer, PreTrainedTokenizerFast), "Complete TODO 2b."
assert tokenizer.pad_token_id is not None and tokenizer.unk_token_id is not None

# %% 3. Inspect an encoded sentence
# TODO 3: Tokenize "A lovely galaxyquokka film"; inspect input_ids,
# attention_mask, and tokenizer.convert_ids_to_tokens(...).
# This made-up word is absent from training and should become [UNK].
encoded_sentence = ...
decoded_tokens = ...
assert encoded_sentence is not Ellipsis and decoded_tokens is not Ellipsis
assert encoded_sentence["input_ids"][0] == tokenizer.cls_token_id
assert encoded_sentence["input_ids"][-1] == tokenizer.sep_token_id
assert tokenizer.unk_token_id in encoded_sentence["input_ids"]
print(decoded_tokens)
# Reflection: how would subword tokenization differ from this WordLevel approach?


# %% 4. Tokenize a Dataset in batches
def tokenize_batch(batch):
    """Return a dictionary of tokenized columns for a batch of text."""
    # TODO 4a: Call tokenizer on batch["text"] with truncation=True and
    # max_length=MAX_LENGTH. Do not pad yet; padding belongs to each model batch.
    raise NotImplementedError("TODO 4a: implement tokenize_batch.")


# TODO 4b: Use splits.map(..., batched=True, remove_columns=["text"]).
# Retain the labels column for the classifier's loss.
tokenized = ...
assert tokenized is not Ellipsis, "Complete TODO 4b."
assert "text" not in tokenized["train"].column_names
assert {"input_ids", "attention_mask", "labels"} <= set(tokenized["train"].column_names)
assert len(tokenized["train"]) == len(splits["train"])
assert max(map(len, tokenized["train"]["input_ids"])) <= MAX_LENGTH

# %% 5. Pad only to the longest example in this batch
# TODO 5: Create DataCollatorWithPadding(tokenizer=..., return_tensors="pt").
# Pass a LIST of BATCH_SIZE individual records from tokenized["train"] to it.
# Dataset[:4] is a dict of columns; [Dataset[i] for i in range(4)] is a list of rows.
collator = ...
batch = ...
assert batch is not Ellipsis, "Complete TODO 5."
assert batch["input_ids"].shape[0] == BATCH_SIZE
assert batch["input_ids"].shape == batch["attention_mask"].shape
assert batch["labels"].shape == (BATCH_SIZE,)
assert batch["input_ids"].dtype == batch["labels"].dtype == torch.long
expected_width = max(len(tokenized["train"][i]["input_ids"]) for i in range(BATCH_SIZE))
assert batch["input_ids"].shape[1] == expected_width
assert torch.all(
    batch["input_ids"][batch["attention_mask"] == 0] == tokenizer.pad_token_id
)
print({key: tuple(value.shape) for key, value in batch.items()})

# %% 6. Build a tiny model with matching vocabulary
config = BertConfig(
    vocab_size=len(tokenizer),
    hidden_size=32,
    num_hidden_layers=1,
    num_attention_heads=2,
    intermediate_size=64,
    max_position_embeddings=64,
    pad_token_id=tokenizer.pad_token_id,
    num_labels=2,
)
# TODO 6: Create BertForSequenceClassification(config).
# Constructing from config initializes random weights and downloads nothing.
model = ...
assert isinstance(model, BertForSequenceClassification), "Complete TODO 6."
assert model.get_input_embeddings().num_embeddings == len(tokenizer)

# %% 7. Inspect the forward output
model.eval()
with torch.no_grad():
    # TODO 7: Pass the entire batch to the model using **batch.
    # Including labels makes the output contain a scalar loss as well as logits.
    outputs = ...
assert outputs is not Ellipsis, "Complete TODO 7."
assert outputs.logits.shape == (BATCH_SIZE, 2)
assert outputs.loss.ndim == 0 and torch.isfinite(outputs.loss)
assert not outputs.logits.requires_grad

# %% 8. Convert logits into probabilities and labels
# TODO 8: Apply softmax over the class dimension; use argmax for predicted ids.
probabilities = ...
predicted_ids = ...
assert probabilities is not Ellipsis and predicted_ids is not Ellipsis
torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(BATCH_SIZE))
assert predicted_ids.shape == (BATCH_SIZE,)
torch.testing.assert_close(predicted_ids, outputs.logits.argmax(dim=1))
print("Random-model class ids:", predicted_ids.tolist())

# %% [markdown]
# ## TODO 9: Explain the representations
#
# - What do input_ids, attention_mask, token_type_ids, and labels represent?
# - Why pad per batch instead of padding every example to MAX_LENGTH?
# - Why must token ids fit the model's embedding vocabulary?
# - Why do plausible-looking probabilities not imply a trained model?
# - Why train a vocabulary only on training text when creating your own tokenizer?
#
# Optional stretch: reuse your PyTorch loop to deliberately overfit the training
# examples. This checks your code; it does not establish generalization.
# Next: 02_pretrained_text_classification.py uses a real pretrained checkpoint.
#
# VS Code practice: inspect batch keys and tensor shapes in the debugger.
# References:
# https://huggingface.co/docs/datasets/process
# https://huggingface.co/docs/transformers/main_classes/data_collator
