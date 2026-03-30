"""
task1_text_generation/models.py

Two language-model architectures for Task 1:
  - LSTMLanguageModel : uses stacked LSTM cells
  - GRULanguageModel  : uses stacked GRU cells

Both share the same constructor signature so they can be swapped in training
scripts with a single argument change.

Architecture overview
─────────────────────
  Embedding  →  Dropout  →  RNN (LSTM / GRU)  →  Dropout  →  Linear  →  logits

The Linear projection maps the hidden state back to vocab_size logits.
Cross-entropy loss is applied externally in the training loop.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------

class _BaseLanguageModel(nn.Module):
    """
    Common functionality shared by LSTM and GRU language models.

    Subclasses only need to set self.rnn and override _init_hidden().
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        embedding: Optional[nn.Embedding] = None,
    ):
        super().__init__()
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # Embedding layer (may be replaced with GloVe / one-hot externally)
        if embedding is not None:
            self.embedding = embedding
            actual_embed_dim = embedding.embedding_dim
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            actual_embed_dim = embed_dim

        self.embed_dropout = nn.Dropout(dropout)
        self.rnn: nn.Module  # set by subclass
        self.rnn_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    # ------------------------------------------------------------------

    def _build_rnn(self, cell_type: str, embed_dim: int, hidden_size: int,
                   num_layers: int, dropout: float):
        """Helper called by subclasses after super().__init__."""
        rnn_cls = nn.LSTM if cell_type == "LSTM" else nn.GRU
        self.rnn = rnn_cls(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(
        self,
        src: Tensor,                   # (batch, seq_len)
        hidden=None,
    ) -> Tuple[Tensor, object]:
        """
        Args:
            src    : token indices  (batch, seq_len)
            hidden : previous hidden state (detached for TBPTT)

        Returns:
            logits : (batch, seq_len, vocab_size)
            hidden : updated hidden state
        """
        emb = self.embed_dropout(self.embedding(src))  # (B, T, E)
        out, hidden = self.rnn(emb, hidden)             # (B, T, H)
        out = self.rnn_dropout(out)
        logits = self.fc(out)                           # (B, T, V)
        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device):
        """Return zero initial hidden state."""
        raise NotImplementedError

    def detach_hidden(self, hidden):
        """Detach hidden state from computation graph (truncated BPTT)."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# LSTM Language Model
# ---------------------------------------------------------------------------

class LSTMLanguageModel(_BaseLanguageModel):
    """
    Multi-layer LSTM language model.

    Parameters
    ──────────
    vocab_size  : size of the token vocabulary
    embed_dim   : embedding dimensionality (ignored when embedding is given)
    hidden_size : number of LSTM hidden units per layer
    num_layers  : number of stacked LSTM layers
    dropout     : dropout probability applied after embedding and RNN output
    embedding   : optional pre-built nn.Embedding (GloVe / one-hot)
    """

    model_name = "LSTM"

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        embedding: Optional[nn.Embedding] = None,
    ):
        super().__init__(vocab_size, embed_dim, hidden_size, num_layers, dropout, embedding)
        actual_dim = self.embedding.embedding_dim
        self._build_rnn("LSTM", actual_dim, hidden_size, num_layers, dropout)
        self.rnn_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self, batch_size: int, device: torch.device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h, c

    def detach_hidden(self, hidden):
        h, c = hidden
        return h.detach(), c.detach()


# ---------------------------------------------------------------------------
# GRU Language Model
# ---------------------------------------------------------------------------

class GRULanguageModel(_BaseLanguageModel):
    """
    Multi-layer GRU language model.

    Parameters
    ──────────
    vocab_size  : size of the token vocabulary
    embed_dim   : embedding dimensionality (ignored when embedding is given)
    hidden_size : number of GRU hidden units per layer
    num_layers  : number of stacked GRU layers
    dropout     : dropout probability applied after embedding and RNN output
    embedding   : optional pre-built nn.Embedding (GloVe / one-hot)
    """

    model_name = "GRU"

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        embedding: Optional[nn.Embedding] = None,
    ):
        super().__init__(vocab_size, embed_dim, hidden_size, num_layers, dropout, embedding)
        actual_dim = self.embedding.embedding_dim
        self._build_rnn("GRU", actual_dim, hidden_size, num_layers, dropout)
        self.rnn_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self, batch_size: int, device: torch.device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

    def detach_hidden(self, hidden):
        return hidden.detach()


# ---------------------------------------------------------------------------
# Basic RNN Language Model (bonus third architecture)
# ---------------------------------------------------------------------------

class RNNLanguageModel(_BaseLanguageModel):
    """
    Vanilla (Elman) RNN language model.

    Included as a performance baseline — expected to underperform LSTM/GRU
    due to vanishing gradients over long sequences.
    """

    model_name = "RNN"

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        embedding: Optional[nn.Embedding] = None,
    ):
        super().__init__(vocab_size, embed_dim, hidden_size, num_layers, dropout, embedding)
        actual_dim = self.embedding.embedding_dim
        self.rnn = nn.RNN(
            input_size=actual_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.rnn_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self, batch_size: int, device: torch.device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

    def detach_hidden(self, hidden):
        return hidden.detach()
