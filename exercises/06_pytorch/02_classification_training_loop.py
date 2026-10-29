"""PyTorch: train and evaluate a small classifier. 60–90 minutes.

Complete 01_tensors_and_autograd first. This worksheet supplies local synthetic
data and the experiment driver; you write preprocessing, the network, training,
evaluation, and best-model selection. No GPU or download is needed.
"""
# ruff: noqa: F401

# %% [markdown]
# # A complete classification experiment
#
# **Goal:** learn a nonlinear decision boundary while keeping training,
# validation, and test responsibilities separate.

# %% 0. Configuration, data, and output directory
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

RANDOM_SEED = 1907
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.01
DEVICE = torch.device("cpu")
torch.manual_seed(RANDOM_SEED)

raw_features, raw_targets = make_moons(
    n_samples=600, noise=0.18, random_state=RANDOM_SEED
)
development_X, test_X, development_y, test_y = train_test_split(
    raw_features,
    raw_targets,
    test_size=0.2,
    stratify=raw_targets,
    random_state=RANDOM_SEED,
)
train_X, validation_X, train_y, validation_y = train_test_split(
    development_X,
    development_y,
    test_size=0.25,
    stratify=development_y,
    random_state=RANDOM_SEED,
)
start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PROJECT_ROOT = next(
    (p for p in [start, *start.parents] if (p / "skills.md").is_file()), None
)
if PROJECT_ROOT is None:
    raise RuntimeError("Run from inside the project.")
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
print("Train/validation/test rows:", len(train_X), len(validation_X), len(test_X))

# %% 1. Preprocess without leakage
# TODO 1: Fit StandardScaler only on train_X; transform all three partitions.
# Convert features to float32 tensors on DEVICE and labels to long tensors.
scaler = StandardScaler()
X_train = torch.tensor(scaler.fit_transform(train_X), dtype=torch.float32, device=DEVICE)
X_validation = torch.tensor(scaler.transform(validation_X), dtype=torch.float32, device=DEVICE)
X_test = torch.tensor(scaler.transform(test_X), dtype=torch.float32, device=DEVICE)
y_train = torch.tensor(train_y, dtype=torch.long, device=DEVICE)
y_validation = torch.tensor(validation_y, dtype=torch.long, device=DEVICE)
y_test = torch.tensor(test_y, dtype=torch.long, device=DEVICE)
assert X_train is not Ellipsis and y_train is not Ellipsis, "Complete TODO 1."
assert X_train.shape == (360, 2)
assert X_validation.shape == X_test.shape == (120, 2)
for inputs, labels in [
    (X_train, y_train),
    (X_validation, y_validation),
    (X_test, y_test),
]:
    assert inputs.dtype == torch.float32 and labels.dtype == torch.long
    assert labels.shape == (len(inputs),)
    assert inputs.device == labels.device == DEVICE
torch.testing.assert_close(X_train.mean(dim=0), torch.zeros(2), atol=1e-6, rtol=0)

# %% 2. Batch the training data
# TODO 2: Wrap X_train/y_train in TensorDataset. Create a DataLoader with
# BATCH_SIZE, shuffle=True, and num_workers=0 for simple interactive execution.

# FIRST TensorDATASET
# SECOND DataLoader to BATCH and ITERATE over the dataset

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
assert isinstance(train_dataset, TensorDataset), "Complete TODO 2."
assert isinstance(train_loader, DataLoader)
batch_X, batch_y = next(iter(train_loader))
assert batch_X.shape == (BATCH_SIZE, 2) and batch_y.shape == (BATCH_SIZE,)

# %% 3. Define the model, loss, and optimizer
# TODO 3: Build nn.Sequential: Linear(2, 16), ReLU, Linear(16, 16), ReLU,
# Linear(16, 2). Move it to DEVICE. Use CrossEntropyLoss and Adam.
# CrossEntropyLoss expects raw logits and integer labels; leave out Softmax.
model = nn.Sequential(
    nn.Linear(2,16),
    nn.ReLU(),
    nn.Linear(16,16),
    nn.ReLU(),
    nn.Linear(16,2)
)
model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
assert isinstance(model, nn.Module), "Complete TODO 3."
assert isinstance(criterion, nn.CrossEntropyLoss)
assert isinstance(optimizer, torch.optim.Adam)
assert model(batch_X).shape == (BATCH_SIZE, 2)


# %% 4. Implement one training epoch
def train_one_epoch(model, loader, optimizer, criterion):
    """Return sample-weighted mean training loss as a Python float."""
    # TODO 4: Set model.train(). For each batch: clear gradients, forward,
    # compute loss, backward, optimizer.step(). Sum loss.item() * batch_size,
    # then divide by the number of examples, not the number of batches.
    # The last batch is smaller, so an unweighted mean of batch means is wrong.
    acc = 0
    acc_n = 0
    for inputs, labels in loader:
        # 1-CLEAR GRADIENTS in OPTIMIZER
        optimizer.zero_grad()
        # 2-FORWARD PASS
        prediction = model(inputs)
        # 3-COMPUTE LOSS
        loss = criterion(prediction, labels)
        # 4-BACKWARD PASS
        loss.backward()
        # 5-UPDATE using OPTIMIZER
        optimizer.step()
        acc += loss.item() * ( inputs.size(0))
        acc_n += inputs.size(0)
    return  acc/acc_n
# %% 5. Implement evaluation
def evaluate(model, features, labels, criterion):
    """Return (mean_loss, accuracy), both Python floats."""
    # TODO 5: Set model.eval() and use torch.no_grad() or torch.inference_mode().
    # Compute loss and accuracy; select labels with logits.argmax(dim=1).
    # Do not update parameters. eval() alone does not disable autograd.
    was_training = model.training
    model.eval() #changes dropout, batchnorm etc, no effect in my model
    with torch.no_grad():
        fpass = model(features)
        loss = criterion(fpass, labels)
        mean_loss = loss.mean().item()
        accu = (fpass.argmax(dim=1) == labels).float().mean().item()
    model.train(was_training)
    return mean_loss, accu

# %% 6. Run epochs and keep the best validation state
history = []
best_validation_loss = float("inf")
best_state = None
for epoch in range(EPOCHS):
    training_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    validation_loss, validation_accuracy = evaluate(
        model, X_validation, y_validation, criterion
    )
    history.append(
        {
            "epoch": epoch + 1,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }
    )
    # TODO 6: If validation_loss improves, update best_validation_loss and
    # set best_state to deepcopy(model.state_dict()). A shallow reference keeps
    # changing as training continues.
    # Write your best-state update here.
    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        best_state = deepcopy(model.state_dict())
        
assert best_state is not None, "Complete TODO 6: keep the best validation state."
assert len(history) == EPOCHS
assert np.isfinite([row["training_loss"] for row in history]).all()
assert 0 <= history[-1]["validation_accuracy"] <= 1
print("Last epoch:", history[-1])

# %% 7. Evaluate the selected model on the test set
# TODO 7: Restore best_state with load_state_dict, then evaluate on X_test/y_test.
# Use validation for model choices; inspect the test result only after selection.
# Write the restore call here.
model.load_state_dict(best_state)
test_loss, test_accuracy = evaluate(model, X_test, y_test, criterion)
assert test_loss is not Ellipsis and test_accuracy is not Ellipsis, "Complete TODO 7."
assert np.isfinite(test_loss) and 0 <= test_accuracy <= 1
print("Held-out test loss/accuracy:", test_loss, test_accuracy)

# %% 8. Plot the learning curves
# TODO 8: Plot training and validation loss against epoch on one Axes.
# Add a title, labeled axes, and a legend. Use tight_layout() and save the figure
# to FIGURE_DIR / "pytorch_learning_curves.png" before calling plt.show().
# Write your plotting and export code here.
fig,ax = plt.subplots()
fig.suptitle("learning curves")
ax.plot([row["epoch"] for row in history], [row["training_loss"] for row in history], label="training loss")
ax.plot([row["epoch"] for row in history], [row["validation_loss"] for row in history], label="validation loss")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.set_xticks(range(1, EPOCHS+1))
ax.legend()
fig.tight_layout()
plt.show()
fig.savefig(FIGURE_DIR / "pytorch_learning_curves.png")
assert (FIGURE_DIR / "pytorch_learning_curves.png").is_file(), "Complete TODO 8."

# %% [markdown]
# ## TODO 9: Interpret the experiment
#
# - Did training and validation loss move together?
# - What is the difference between model.eval() and disabling gradient tracking?
# - Why did you copy the best state instead of storing model.state_dict() directly?
# - Where would preprocessing leakage or test-set tuning enter this workflow?
#
# Optional stretch: vary width or learning rate using validation results. Plot a
# decision boundary. After inspecting the test set, use a fresh holdout for any
# unbiased assessment of further experiments.
#
# VS Code practice: set a breakpoint inside your training loop and inspect logits,
# labels, and gradients. Keep DataLoader workers at 0 while debugging.
# Reference: https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
