"""
task2_machine_translation/models.py

Seq2Seq encoder-decoder architectures for machine translation.

Two RNN cell types are provided (LSTM and GRU) that share the same
encoder-decoder API so they can be swapped in one line.

Architecture
────────────

              ┌──────────┐           ┌──────────┐
  src_ids ──▶ │ Encoder  │ ──state──▶ │ Decoder  │ ──▶ tgt_logits
              └──────────┘           └──────────┘
                                         ▲
                                      tgt_ids (shifted)

During training  : teacher forcing with probability `teacher_forcing_ratio`.
During inference : greedy decoding token-by-token.

References
──────────
Sutskever et al. (2014) "Sequence to Sequence Learning with Neural Networks"
Cho et al.       (2014) "Learning Phrase Representations using RNN Encoder-Decoder"
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    RNN encoder that compresses a variable-length source sequence into a
    fixed-size context vector (the final hidden state).

    Args:
        vocab_size  : source vocabulary size
        embed_dim   : embedding dimension
        hidden_size : RNN hidden size
        num_layers  : number of stacked RNN layers
        dropout     : dropout probability
        cell_type   : "LSTM" | "GRU"
        embedding   : optional pre-built nn.Embedding (GloVe / one-hot)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        cell_type: str = "LSTM",
        embedding: Optional[nn.Embedding] = None,
    ):
        super().__init__()
        self.cell_type = cell_type

        if embedding is not None:
            self.embedding = embedding
            actual_dim = embedding.embedding_dim
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            actual_dim = embed_dim

        self.dropout = nn.Dropout(dropout)
        rnn_cls = nn.LSTM if cell_type == "LSTM" else nn.GRU
        self.rnn = rnn_cls(
            actual_dim, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, src: Tensor):
        """
        Args:
            src: (batch, src_len) — padded source token ids
        Returns:
            outputs : (batch, src_len, hidden_size)
            hidden  : final hidden state (passed to decoder)
        """
        emb = self.dropout(self.embedding(src))   # (B, T, E)
        outputs, hidden = self.rnn(emb)            # (B, T, H), state
        return outputs, hidden


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """
    RNN decoder with optional teacher forcing.

    Processes one token per step, using the encoder's final hidden state as
    the initial hidden state.

    Args:
        vocab_size  : target vocabulary size
        embed_dim   : embedding dimension
        hidden_size : RNN hidden size (must match encoder's hidden_size)
        num_layers  : number of stacked RNN layers
        dropout     : dropout probability
        cell_type   : "LSTM" | "GRU"
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        cell_type: str = "LSTM",
    ):
        super().__init__()
        self.cell_type  = cell_type
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout   = nn.Dropout(dropout)
        rnn_cls = nn.LSTM if cell_type == "LSTM" else nn.GRU
        self.rnn = rnn_cls(
            embed_dim, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward_step(self, token: Tensor, hidden):
        """
        One decoding step.

        Args:
            token  : (batch,) — previous output token id
            hidden : decoder hidden state
        Returns:
            logits : (batch, vocab_size)
            hidden : updated hidden state
        """
        emb = self.dropout(self.embedding(token.unsqueeze(1)))  # (B, 1, E)
        out, hidden = self.rnn(emb, hidden)                      # (B, 1, H)
        logits = self.fc_out(out.squeeze(1))                     # (B, V)
        return logits, hidden


# ---------------------------------------------------------------------------
# Seq2Seq wrapper
# ---------------------------------------------------------------------------

class Seq2Seq(nn.Module):
    """
    Full encoder-decoder model.

    Args:
        encoder              : Encoder instance
        decoder              : Decoder instance
        tgt_bos_idx          : BOS token index for initialising decoding
        tgt_eos_idx          : EOS token index (used during inference)
        teacher_forcing_ratio: probability of using ground-truth token at each step
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        tgt_bos_idx: int,
        tgt_eos_idx: int,
        teacher_forcing_ratio: float = 0.5,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_bos_idx = tgt_bos_idx
        self.tgt_eos_idx = tgt_eos_idx
        self.teacher_forcing_ratio = teacher_forcing_ratio

        assert encoder.rnn.hidden_size == decoder.rnn.hidden_size, \
            "Encoder and decoder must have the same hidden_size"
        assert encoder.cell_type == decoder.cell_type, \
            "Encoder and decoder must use the same RNN cell type"

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """
        Teacher-forced forward pass used during training.

        Args:
            src : (batch, src_len)
            tgt : (batch, tgt_len)  — includes BOS, excludes final EOS

        Returns:
            all_logits : (batch, tgt_len-1, tgt_vocab_size)
        """
        batch_size = src.size(0)
        tgt_len    = tgt.size(1)
        vocab_size = self.decoder.vocab_size

        _, hidden = self.encoder(src)

        # Start with BOS token
        dec_input = tgt[:, 0]          # (B,) — should all be BOS_IDX
        all_logits = []

        for t in range(1, tgt_len):
            logits, hidden = self.decoder.forward_step(dec_input, hidden)
            all_logits.append(logits.unsqueeze(1))

            # Teacher forcing
            use_teacher = random.random() < self.teacher_forcing_ratio
            if use_teacher:
                dec_input = tgt[:, t]
            else:
                dec_input = logits.argmax(dim=-1)

        return torch.cat(all_logits, dim=1)   # (B, T-1, V)

    # ------------------------------------------------------------------
    # Greedy inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def translate(
        self,
        src: Tensor,
        max_len: int = 50,
        device=None,
    ) -> Tensor:
        """
        Greedy decoding — returns predicted token ids for each sentence.

        Args:
            src     : (batch, src_len)
            max_len : maximum number of output tokens

        Returns:
            predictions : (batch, max_len) — token ids, padded with EOS
        """
        batch_size = src.size(0)
        _, hidden  = self.encoder(src)

        dec_input = torch.full(
            (batch_size,), self.tgt_bos_idx, dtype=torch.long, device=src.device
        )
        predictions = []
        finished    = torch.zeros(batch_size, dtype=torch.bool, device=src.device)

        for _ in range(max_len):
            logits, hidden = self.decoder.forward_step(dec_input, hidden)
            pred = logits.argmax(dim=-1)          # (B,)
            predictions.append(pred.unsqueeze(1))
            finished = finished | (pred == self.tgt_eos_idx)
            if finished.all():
                break
            dec_input = pred

        return torch.cat(predictions, dim=1)      # (B, steps)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_seq2seq(
    src_vocab_size: int,
    tgt_vocab_size: int,
    tgt_bos_idx: int,
    tgt_eos_idx: int,
    embed_dim: int = 256,
    hidden_size: int = 256,
    num_layers: int = 2,
    dropout: float = 0.3,
    cell_type: str = "LSTM",
    src_embedding: Optional[nn.Embedding] = None,
    teacher_forcing_ratio: float = 0.5,
) -> Seq2Seq:
    """
    Convenience factory for building a full Seq2Seq model.

    Args:
        src_embedding : optional pre-built embedding for the source language
        cell_type     : "LSTM" | "GRU"
    """
    encoder = Encoder(
        vocab_size  = src_vocab_size,
        embed_dim   = embed_dim,
        hidden_size = hidden_size,
        num_layers  = num_layers,
        dropout     = dropout,
        cell_type   = cell_type,
        embedding   = src_embedding,
    )
    # Decoder always uses trainable embeddings (target language)
    decoder = Decoder(
        vocab_size  = tgt_vocab_size,
        embed_dim   = embed_dim,
        hidden_size = hidden_size,
        num_layers  = num_layers,
        dropout     = dropout,
        cell_type   = cell_type,
    )
    return Seq2Seq(encoder, decoder, tgt_bos_idx, tgt_eos_idx, teacher_forcing_ratio)
