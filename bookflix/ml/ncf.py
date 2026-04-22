"""
Neural Collaborative Filtering (NCF) — Section 2.3 of thesis.

Replaces matrix factorization (SVD) with a nonlinear MLP that models
user–item interactions through learned embeddings, capturing complex
preference patterns that SVD cannot represent.
"""

import torch
import torch.nn as nn


class NCF(nn.Module):
    """
    Two-path MLP neural collaborative filter.
    Concatenates user and item embeddings, then passes them through
    a stack of fully connected hidden layers to predict a rating.
    """

    def __init__(self, num_users, num_items, embedding_dim=32,
                 hidden_layers=None, dropout=0.2):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [64, 32, 16]

        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)

        layers = []
        in_dim = embedding_dim * 2
        for out_dim in hidden_layers:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        u = self.user_embedding(user_ids)
        v = self.item_embedding(item_ids)
        x = torch.cat([u, v], dim=-1)
        return self.mlp(x).squeeze(-1)

    def get_user_embedding(self, user_idx: int) -> torch.Tensor:
        with torch.no_grad():
            return self.user_embedding(torch.tensor([user_idx]))

    def get_item_embedding(self, item_idx: int) -> torch.Tensor:
        with torch.no_grad():
            return self.item_embedding(torch.tensor([item_idx]))
