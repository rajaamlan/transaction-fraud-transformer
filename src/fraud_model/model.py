from __future__ import annotations

import torch
from torch import nn


class TransactionTransformer(nn.Module):
    """Transformer encoder for mixed numeric and categorical transaction rows."""

    def __init__(
        self,
        num_numeric_features: int,
        category_cardinalities: list[int],
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.numeric_weight = nn.Parameter(torch.randn(num_numeric_features, d_model) * 0.02)
        self.numeric_bias = nn.Parameter(torch.zeros(num_numeric_features, d_model))
        self.category_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, d_model) for cardinality in category_cardinalities]
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        numeric_tokens = numeric.unsqueeze(-1) * self.numeric_weight.unsqueeze(0) + self.numeric_bias.unsqueeze(0)
        categorical_tokens = [
            embedding(categorical[:, idx]) for idx, embedding in enumerate(self.category_embeddings)
        ]
        tokens = [numeric_tokens, *[token.unsqueeze(1) for token in categorical_tokens]]
        sequence = torch.cat(tokens, dim=1)
        cls = self.cls_token.expand(sequence.size(0), -1, -1)
        encoded = self.encoder(torch.cat([cls, sequence], dim=1))
        return self.classifier(encoded[:, 0]).squeeze(-1)
