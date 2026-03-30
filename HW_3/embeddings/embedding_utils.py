"""
embeddings/embedding_utils.py

Provides two embedding strategies required by the assignment:
  1. Pre-trained GloVe embeddings  (50-d downloaded on demand)
  2. One-hot encoding              (sparse, no semantic information)

Both return a torch.nn.Embedding layer that can be plugged directly into
any model defined in this project.
"""

import os
import math
import zipfile
import urllib.request
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# GloVe helpers
# ---------------------------------------------------------------------------

GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_DIM = 50  # use 50-d vectors for speed; swap to 100/200/300 if desired


def _download_glove(cache_dir: str = "data/glove") -> str:
    """
    Download GloVe 6B vectors if not already present.
    Returns the path to the 50-d text file.
    """
    os.makedirs(cache_dir, exist_ok=True)
    txt_path = os.path.join(cache_dir, f"glove.6B.{GLOVE_DIM}d.txt")
    if os.path.exists(txt_path):
        return txt_path

    zip_path = os.path.join(cache_dir, "glove.6B.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading GloVe 6B vectors to {zip_path} …")
        urllib.request.urlretrieve(GLOVE_URL, zip_path)

    print("Extracting GloVe …")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extract(f"glove.6B.{GLOVE_DIM}d.txt", cache_dir)

    return txt_path


def load_glove_vectors(cache_dir: str = "data/glove") -> Dict[str, np.ndarray]:
    """
    Load GloVe word vectors into a plain Python dict  {word -> np.ndarray}.
    Downloads the file on first call.
    """
    txt_path = _download_glove(cache_dir)
    vectors: Dict[str, np.ndarray] = {}
    print(f"Loading GloVe from {txt_path} …")
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            vectors[word] = vec
    print(f"  Loaded {len(vectors):,} GloVe vectors (dim={GLOVE_DIM})")
    return vectors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_glove_embedding(
    vocab: Dict[str, int],
    glove_vectors: Optional[Dict[str, np.ndarray]] = None,
    freeze: bool = False,
    cache_dir: str = "data/glove",
) -> nn.Embedding:
    """
    Build an nn.Embedding whose weight matrix is initialised from GloVe.

    Words not found in GloVe are initialised with a small random vector so
    training can still update their representations.

    Args:
        vocab        : {word: index} mapping built from your dataset.
        glove_vectors: pre-loaded GloVe dict (loaded once externally for speed).
        freeze       : if True the embedding weights are not updated during training.
        cache_dir    : where to cache the downloaded GloVe file.

    Returns:
        nn.Embedding of shape (vocab_size, GLOVE_DIM).
    """
    if glove_vectors is None:
        glove_vectors = load_glove_vectors(cache_dir)

    vocab_size = len(vocab)
    weight = np.zeros((vocab_size, GLOVE_DIM), dtype=np.float32)

    found = 0
    for word, idx in vocab.items():
        if word in glove_vectors:
            weight[idx] = glove_vectors[word]
            found += 1
        else:
            # Xavier-like initialisation for OOV words
            scale = math.sqrt(6.0 / GLOVE_DIM)
            weight[idx] = np.random.uniform(-scale, scale, GLOVE_DIM).astype(np.float32)

    coverage = 100.0 * found / max(vocab_size, 1)
    print(f"  GloVe coverage: {found}/{vocab_size} words ({coverage:.1f} %)")

    embedding = nn.Embedding(vocab_size, GLOVE_DIM)
    embedding.weight = nn.Parameter(torch.from_numpy(weight), requires_grad=not freeze)
    return embedding


def build_onehot_embedding(
    vocab_size: int,
    freeze: bool = True,
) -> nn.Embedding:
    """
    Build an nn.Embedding whose weight is a fixed identity matrix (one-hot).

    The embedding dimension equals vocab_size, which is memory-intensive for
    large vocabularies; limit vocab_size to ≤ 5 000 in experiments.

    Args:
        vocab_size : number of tokens in the vocabulary.
        freeze     : should always be True for genuine one-hot encoding.

    Returns:
        nn.Embedding of shape (vocab_size, vocab_size).
    """
    weight = torch.eye(vocab_size)  # identity matrix = one-hot matrix
    embedding = nn.Embedding(vocab_size, vocab_size)
    embedding.weight = nn.Parameter(weight, requires_grad=not freeze)
    return embedding


def build_trainable_embedding(
    vocab_size: int,
    embed_dim: int,
) -> nn.Embedding:
    """
    Randomly initialised trainable embedding (baseline / fallback).

    Not required by the assignment but useful for ablation studies.
    """
    return nn.Embedding(vocab_size, embed_dim, padding_idx=0)
