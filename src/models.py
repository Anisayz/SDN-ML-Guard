"""
models.py — Shared model definitions
======================================
Imported by train.py, evaluate.py, and predict.py.
Keeping Autoencoder and AutoencoderWrapper here ensures pickle can always
find the class definitions regardless of which script loads anomaly.pkl.
"""

import numpy as np
import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """
    Symmetric autoencoder for anomaly detection.
    Encoder compresses input → bottleneck.
    Decoder reconstructs input from bottleneck.

    Anomaly score = weighted MSE(input, reconstruction), where feature
    weights come from RF feature importances — features the RF considers
    most discriminative get higher reconstruction penalty, forcing the
    autoencoder to learn them precisely and fail harder on attack flows
    that violate those important feature patterns.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        feature_weights: np.ndarray = None,  # shape (input_dim,)
    ) -> None:
        super().__init__()

        # Encoder: input_dim → hidden_dims[0] → hidden_dims[1] → ...
        enc_layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev_dim, h), nn.ReLU()]
            prev_dim = h
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder: mirrors encoder in reverse
        dec_layers = []
        for h in reversed(hidden_dims[:-1]):
            dec_layers += [nn.Linear(prev_dim, h), nn.ReLU()]
            prev_dim = h
        dec_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        # Feature weights — stored as a non-trainable buffer so they
        # move to the correct device automatically with model.to(device)
        if feature_weights is not None:
            w = torch.tensor(feature_weights, dtype=torch.float32)
        else:
            w = torch.ones(input_dim, dtype=torch.float32)
        # Normalise so weights sum to input_dim (keeps loss scale stable)
        w = w / w.mean()
        self.register_buffer("feature_weights", w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-sample weighted MSE between input and reconstruction.
        Shape: (n,)

        weighted_error_i = mean(weights * (x_i - recon_i)^2)

        High-importance features (large weight) contribute more to the
        error score — the model is penalised more for failing to
        reconstruct them, which is exactly what we want for attack detection.
        """
        with torch.no_grad():
            recon = self.forward(x)
            sq_err = (x - recon) ** 2                      # (n, input_dim)
            weighted = sq_err * self.feature_weights        # broadcast weights
            return weighted.mean(dim=1)                     # (n,)

    def weighted_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Training loss — weighted MSE, scalar.
        Called during the training loop instead of nn.MSELoss().
        """
        recon  = self.forward(x)
        sq_err = (x - recon) ** 2
        return (sq_err * self.feature_weights).mean()


class AutoencoderWrapper:
    """
    Wraps the PyTorch Autoencoder so predict.py / evaluate.py can call
    wrapper.score_samples(X) the same way IsolationForest did.

    score_samples() returns NEGATIVE reconstruction error so that:
      - more negative  = more anomalous  (same convention as IsolationForest)
      - threshold logic stays: flag if score < threshold
    """

    def __init__(self, model: Autoencoder, device: str, threshold: float) -> None:
        self.model     = model
        self.device    = device
        self.threshold = threshold

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        X: numpy array shape (n, n_features), already scaled with benign_scaler.
        Returns: numpy array shape (n,) — negative weighted reconstruction error.
        """
        self.model.eval()
        tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        errors = self.model.reconstruction_error(tensor).cpu().numpy()
        return -errors   # negate: higher error → more negative → IF convention