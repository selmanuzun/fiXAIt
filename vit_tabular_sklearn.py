# vit_tabular_sklearn.py
# pip install torch skorch

from __future__ import annotations
import numpy as np

import torch
import torch.nn as nn

from skorch import NeuralNetClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


class TabularViT(nn.Module):
    def __init__(self, n_features, n_classes, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.feature_embed = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.feature_embed.unsqueeze(0)  # (B,F,D)
        cls = self.cls_token.expand(x.size(0), -1, -1)              # (B,1,D)
        z = torch.cat([cls, tokens], dim=1)                         # (B,1+F,D)
        z = self.encoder(z)
        z = self.norm(z)
        return self.head(z[:, 0, :])                                # CLS -> logits


class AutoTabularViTClassifier(BaseEstimator, ClassifierMixin):
    """
    Sklearn uyumlu wrapper.
    n_features ve n_classes fit sırasında X,y'den otomatik çıkarılır.
    İçeride skorch NeuralNetClassifier kullanır.
    """

    def __init__(
        self,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
        max_epochs=50,
        lr=1e-3,
        batch_size=64,
        weight_decay=1e-4,
        seed=42,
        device=None,
        verbose=0,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.max_epochs = max_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = device
        self.verbose = verbose

        self.net_ = None  # fit sırasında oluşacak

    def _build_net(self, n_features, n_classes):
        torch.manual_seed(self.seed)
        device = self.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.net_ = NeuralNetClassifier(
            module=TabularViT,
            module__n_features=int(n_features),
            module__n_classes=int(n_classes),
            module__d_model=self.d_model,
            module__n_heads=self.n_heads,
            module__n_layers=self.n_layers,
            module__dropout=self.dropout,
            max_epochs=self.max_epochs,
            lr=self.lr,
            optimizer=torch.optim.AdamW,
            optimizer__weight_decay=self.weight_decay,
            criterion=nn.CrossEntropyLoss,
            batch_size=self.batch_size,
            iterator_train__shuffle=True,
            device=device,
            verbose=self.verbose,
        )

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        self.classes_ = np.unique(y)          # <-- EKLE (kritik)
        n_features = X.shape[1]
        n_classes = self.classes_.shape[0]    # <-- istersen bunu kullan

        self._build_net(n_features, n_classes)
        self.net_.fit(X, y)
        return self


    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        return self.net_.predict(X)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        return self.net_.predict_proba(X)

    def score(self, X, y):
        # sklearn default score: accuracy
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))


def make_vit_classifier(**kwargs):
    """
    Ana kodda: pick_mdl = make_vit_classifier()
    X,y bilinmese de olur; fit sırasında otomatik boyutlandırır.
    """
    return AutoTabularViTClassifier(**kwargs)
