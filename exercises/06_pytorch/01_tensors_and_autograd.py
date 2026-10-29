"""PyTorch: tensors, gradients, and one learning step. 30–45 minutes.

Use Run Cell in VS Code. Replace ... and write code beneath each numbered TODO.
Checkpoints deliberately stop at unfinished tasks. Restart and run from the top
when repeating this exercise: gradients and parameter updates change state.
"""

# %% [markdown]
# # Tensors and autograd
#
# **Goal:** understand how a small linear model computes predictions, a loss,
# gradients, and a parameter update before using a training loop.

# %% 0. Supplied inputs — three observations, two features
import numpy as np
import torch

RANDOM_SEED = 42
LEARNING_RATE = 0.01
torch.manual_seed(RANDOM_SEED)

features = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
targets = torch.tensor([2.0, 4.0, 6.0])
print(features.shape, features.dtype, features.device)

# %% 1. Create trainable parameters
# TODO 1: Create a floating-point weights tensor with values [0.5, -0.25]
# and a scalar bias tensor equal to 0.1. Enable requires_grad on both.
# Hint: torch.tensor(..., requires_grad=True).
weights = torch.tensor([0.5, -0.25], requires_grad=True)
bias = torch.tensor(0.1, requires_grad=True)
assert weights is not Ellipsis and bias is not Ellipsis, "Complete TODO 1."
assert weights.shape == (2,) and bias.shape == torch.Size([])
assert weights.requires_grad and bias.requires_grad
assert weights.is_leaf and bias.is_leaf
torch.testing.assert_close(weights.detach(), torch.tensor([0.5, -0.25]))
torch.testing.assert_close(bias.detach(), torch.tensor(0.1))

# %% 2. Make predictions
# TODO 2: Compute features @ weights + bias.
# Predict each intermediate shape on paper before running the expression.
predictions = features @ weights + bias
assert predictions is not Ellipsis, "Complete TODO 2."
assert predictions.shape == targets.shape
torch.testing.assert_close(predictions, torch.tensor([0.1, 0.6, 1.1]))

# %% 3. Build a scalar loss
# TODO 3: Compute mean squared error manually using tensor operations.
# Reduce across observations; do not detach predictions while computing the loss.
loss = ((predictions - targets) ** 2).mean()
assert loss is not Ellipsis, "Complete TODO 3."
assert loss.ndim == 0 and loss.requires_grad
torch.testing.assert_close(loss.detach(), torch.tensor(13.06))
initial_loss = loss.item()

# %% 4. Compare automatic and manual derivatives
# TODO 4: Call loss.backward(), then inspect weights.grad and bias.grad.
# Also calculate the analytical gradients:
# dL/dw = (2/n) X^T (prediction - target)
# dL/db = (2/n) sum(prediction - target)
# Write the backward call here.
loss.backward()
n = targets.shape[0]
manual_weight_gradient = (2/n)*features.T @ (predictions-targets)
manual_bias_gradient = (2/n)*((predictions - targets).sum())
assert weights.grad is not None and bias.grad is not None, "Run backward."
assert manual_weight_gradient is not Ellipsis and manual_bias_gradient is not Ellipsis
torch.testing.assert_close(weights.grad, manual_weight_gradient)
torch.testing.assert_close(bias.grad, manual_bias_gradient)

# %% 5. Observe gradient accumulation
first_weight_gradient = weights.grad.detach().clone()
first_bias_gradient = bias.grad.detach().clone()
# TODO 5: Recompute the same forward pass and loss, then call backward again
# without clearing gradients. Build a new graph; do not reuse loss.backward().
second_loss = ((features @ weights + bias - targets) ** 2).mean()
second_loss.backward() 
assert second_loss is not Ellipsis, "Complete TODO 5."
# Write the second backward call here.
torch.testing.assert_close(weights.grad, 2 * first_weight_gradient)
torch.testing.assert_close(bias.grad, 2 * first_bias_gradient)

# %% 6. Take exactly one gradient-descent step
weights_before = weights.detach().clone()
# TODO 6:
# - Clear both gradients by assigning .grad = None.
# - Recompute the loss and call backward once.
# - Inside torch.no_grad(), subtract LEARNING_RATE * gradient from each parameter.
# Why does the update belong outside the gradient-tracking context?
# Write your step here.
weights.grad = None
bias.grad = None
loss = ((features @ weights + bias - targets) ** 2).mean()
loss.backward()

# UPDATE parameters = STEP INSIDE a torch.no_grad()
with torch.no_grad():
    weights -= LEARNING_RATE * weights.grad
    bias -= LEARNING_RATE * bias.grad
assert not torch.equal(weights.detach(), weights_before), "Complete TODO 6."
torch.testing.assert_close(
    weights.detach(), weights_before - LEARNING_RATE * first_weight_gradient
)
torch.testing.assert_close(
    bias.detach(), torch.tensor(0.1) - LEARNING_RATE * first_bias_gradient
)
assert weights.is_leaf and bias.is_leaf

# %% 7. Evaluate and cross the NumPy boundary
# TODO 7: Compute fresh predictions and MSE after the update. Store the loss as
# a Python float and convert the predictions to a NumPy array.
# Hint: .item() and .detach().cpu().numpy().
updated_predictions = (features @ weights + bias)
updated_loss = ((updated_predictions - targets) ** 2).mean().item()
numpy_predictions = updated_predictions.detach().numpy()
assert updated_predictions is not Ellipsis and updated_loss is not Ellipsis
assert isinstance(updated_loss, float) and updated_loss < initial_loss
assert isinstance(numpy_predictions, np.ndarray) and numpy_predictions.shape == (3,)
np.testing.assert_allclose(numpy_predictions, updated_predictions.detach().numpy())
print("Loss before/after:", initial_loss, updated_loss)

# %% [markdown]
# ## TODO 8: Explain what happened
#
# - What did requires_grad change, and where were gradients stored?
# - Why did the second backward pass add to the first gradient?
# - How do no_grad(), detach(), and clearing .grad differ?
# - What happens if targets has shape (3, 1) while predictions has shape (3,)?
#   Try this separately and inspect the residual shape before taking its mean.
#
# Optional stretch: turn the single update into 50 steps and plot loss. Try a
# larger learning rate and explain any divergence.
#
# VS Code practice: inspect .shape, .grad, and .grad_fn at a breakpoint.
# Reference: https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
