"""
Lasso regression model and training utilities.
"""

import torch
import torch.nn as nn


class LassoRegression(nn.Module):
    """
    Lasso regression model with L1 regularization.
    """

    def __init__(self, n_features, lam=1e-3):
        """
        Initialize Lasso regression model.

        Args:
            n_features: Number of input features
            lam: L1 regularization strength (default: 1e-3)
        """
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_features))
        self.b = nn.Parameter(torch.zeros(1))
        self.lam = lam

    def forward(self, X):
        """
        Forward pass.

        Args:
            X: Input tensor of shape (batch, n_features)

        Returns:
            Output tensor of shape (batch,)
        """
        return X @ self.w + self.b


def soft_thresholding(w, alpha):
    """
    Soft thresholding operator for L1 regularization.

    Args:
        w: Weight tensor
        alpha: Threshold value

    Returns:
        Soft-thresholded weights
    """
    return torch.sign(w) * torch.clamp(torch.abs(w) - alpha, min=0)


def train_lasso(model, X, y, lr=1e-2, steps=5000):
    """
    Train a Lasso regression model.

    Args:
        model: LassoRegression model instance
        X: Input tensor of shape (batch, n_features)
        y: Target tensor of shape (batch,)
        lr: Learning rate (default: 1e-2)
        steps: Number of training steps (default: 5000)

    Returns:
        Trained model
    """
    for _ in range(steps):
        # Gradient step on MSE
        y_pred = model(X)
        loss = ((y - y_pred) ** 2).mean()
        loss.backward()

        # Update weights
        with torch.no_grad():
            model.w -= lr * model.w.grad
            model.b -= lr * model.b.grad

            # L1 proximal operator (soft thresholding)
            model.w[:] = soft_thresholding(model.w, model.lam * lr)

        model.w.grad.zero_()
        model.b.grad.zero_()

    return model
