"""
Neural Collaborative Filtering (NCF) — Section 2.3 of thesis.

Replaces matrix factorization (SVD) with a nonlinear MLP that models
user–item interactions through learned embeddings, capturing complex
preference patterns that SVD cannot represent.

Cold-start extension (Section 2.4):
  infer_user_embedding() computes a pseudo user embedding as a weighted
  mean of item embeddings for the books a new user has already rated.
  This allows the full NCF hybrid to be applied even when the user has
  no entry in the training data — no retraining required.
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

    # ------------------------------------------------------------------
    # Стандартний прямий прохід (для відомих юзерів)
    # ------------------------------------------------------------------

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        u = self.user_embedding(user_ids)
        v = self.item_embedding(item_ids)
        x = torch.cat([u, v], dim=-1)
        return self.mlp(x).squeeze(-1)

    # ------------------------------------------------------------------
    # Cold-start: інференс ембедінгу нового юзера (Section 2.4)
    # ------------------------------------------------------------------

    def infer_user_embedding(
        self,
        item_ratings: list[tuple[int, float]],
    ) -> torch.Tensor:
        """
        Виводить псевдо-ембедінг нового користувача як зважене середнє
        ембедінгів книг, які він оцінив.

        item_ratings: [(item_idx, rating), ...]
        Повертає: тензор форми (1, embedding_dim)
        """
        if not item_ratings:
            emb_dim = self.user_embedding.embedding_dim
            return torch.zeros(1, emb_dim)

        with torch.no_grad():
            item_idxs = torch.tensor(
                [idx for idx, _ in item_ratings], dtype=torch.long
            )
            ratings = torch.tensor(
                [float(r) for _, r in item_ratings], dtype=torch.float32
            )
            item_embs = self.item_embedding(item_idxs)   # (n, emb_dim)

            # Нормалізуємо ваги за рейтингом
            weights = (ratings / ratings.sum()).unsqueeze(1)   # (n, 1)
            user_emb = (item_embs * weights).sum(dim=0, keepdim=True)  # (1, emb_dim)

        return user_emb

    def forward_with_inferred_embedding(
        self,
        user_emb: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Прямий прохід із зовнішнім (виведеним) ембедінгом юзера.
        user_emb : (1, embedding_dim)
        item_ids : (n,)
        """
        item_embs = self.item_embedding(item_ids)               # (n, emb_dim)
        user_embs = user_emb.expand(item_ids.shape[0], -1)     # (n, emb_dim)
        x = torch.cat([user_embs, item_embs], dim=-1)
        return self.mlp(x).squeeze(-1)

    def get_user_embedding(self, user_idx: int) -> torch.Tensor:
        with torch.no_grad():
            return self.user_embedding(torch.tensor([user_idx]))

    def get_item_embedding(self, item_idx: int) -> torch.Tensor:
        with torch.no_grad():
            return self.item_embedding(torch.tensor([item_idx]))
