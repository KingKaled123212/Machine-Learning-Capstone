"""
task1_text_generation/data_loader.py

Loads the WikiText-2 dataset (or falls back to a built-in Shakespeare excerpt)
and builds a character / word-level vocabulary for language-model training.

Two datasets are supported so the assignment works even without internet:
  - WikiText-2  (downloaded via torchtext or a direct URL)
  - Shakespeare (a short excerpt bundled inline)

The module exposes:
  - Vocabulary  : simple bidirectional word ↔ index mapping
  - TextDataset : torch.utils.data.Dataset that yields (input_seq, target_seq) pairs
  - get_dataloaders() : convenience function returning train / val / test loaders
"""

import os
import re
import urllib.request
from typing import List, Tuple, Optional, Dict

import torch
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Built-in Shakespeare excerpt (fallback when network is unavailable)
# ---------------------------------------------------------------------------
SHAKESPEARE_TEXT = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil,
Must give us pause.
All the world's a stage, and all the men and women merely players;
They have their exits and their entrances,
And one man in his time plays many parts,
His acts being seven ages.
Friends, Romans, countrymen, lend me your ears;
I come to bury Caesar, not to praise him.
The evil that men do lives after them;
The good is oft interred with their bones.
What a piece of work is a man! How noble in reason, how infinite in faculty!
In form and moving how express and admirable! In action how like an angel!
In apprehension how like a god! The beauty of the world. The paragon of animals.
""" * 20  # repeat to get a reasonably-sized corpus


WIKITEXT2_TRAIN_URL = (
    "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"
)
WIKITEXT2_VALID_URL = (
    "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt"
)
WIKITEXT2_TEST_URL = (
    "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt"
)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class Vocabulary:
    """Bidirectional word ↔ integer mapping."""

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"

    def __init__(self, min_freq: int = 2, max_size: Optional[int] = None):
        self.min_freq = min_freq
        self.max_size = max_size
        self._word2idx: Dict[str, int] = {}
        self._idx2word: Dict[int, str] = {}

    # --- build ----------------------------------------------------------------

    def build(self, tokens: List[str]) -> "Vocabulary":
        from collections import Counter

        freq = Counter(tokens)
        specials = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]

        candidates = [w for w, c in freq.most_common() if c >= self.min_freq]
        if self.max_size:
            candidates = candidates[: self.max_size - len(specials)]

        for i, tok in enumerate(specials + candidates):
            self._word2idx[tok] = i
            self._idx2word[i] = tok
        return self

    # --- properties -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._word2idx)

    @property
    def pad_idx(self) -> int:
        return self._word2idx[self.PAD_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self._word2idx[self.UNK_TOKEN]

    # --- encode / decode ------------------------------------------------------

    def encode(self, tokens: List[str]) -> List[int]:
        unk = self.unk_idx
        return [self._word2idx.get(t, unk) for t in tokens]

    def decode(self, indices: List[int]) -> List[str]:
        return [self._idx2word.get(i, self.UNK_TOKEN) for i in indices]

    def word2idx(self, word: str) -> int:
        return self._word2idx.get(word, self.unk_idx)

    def idx2word(self, idx: int) -> str:
        return self._idx2word.get(idx, self.UNK_TOKEN)


# ---------------------------------------------------------------------------
# Text tokenisation
# ---------------------------------------------------------------------------

def simple_tokenise(text: str) -> List[str]:
    """Lower-case and split on whitespace / punctuation."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return text.split()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    """
    Sliding-window language-model dataset.

    Each sample is (input_seq, target_seq) where target_seq is input_seq
    shifted one step to the right (next-word prediction).
    """

    def __init__(self, token_ids: List[int], seq_len: int = 30):
        self.seq_len = seq_len
        # Trim so length is a multiple of seq_len
        total = (len(token_ids) // seq_len) * seq_len
        data = torch.tensor(token_ids[:total], dtype=torch.long)
        # Reshape to (num_sequences, seq_len)
        self.inputs = data[:-1].unfold(0, seq_len, seq_len)
        self.targets = data[1:].unfold(0, seq_len, seq_len)
        # Trim to same length
        n = min(len(self.inputs), len(self.targets))
        self.inputs = self.inputs[:n]
        self.targets = self.targets[:n]

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


# ---------------------------------------------------------------------------
# Download / load raw text
# ---------------------------------------------------------------------------

def _fetch_wikitext2(data_dir: str = "data/wikitext2") -> Tuple[str, str, str]:
    os.makedirs(data_dir, exist_ok=True)
    splits = {
        "train": (WIKITEXT2_TRAIN_URL, os.path.join(data_dir, "train.txt")),
        "valid": (WIKITEXT2_VALID_URL, os.path.join(data_dir, "valid.txt")),
        "test":  (WIKITEXT2_TEST_URL,  os.path.join(data_dir, "test.txt")),
    }
    texts = {}
    for split, (url, path) in splits.items():
        if not os.path.exists(path):
            print(f"  Downloading WikiText-2 {split} split …")
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                print(f"  Download failed ({e}); using Shakespeare fallback.")
                return None, None, None
        with open(path, encoding="utf-8") as f:
            texts[split] = f.read()
    return texts["train"], texts["valid"], texts["test"]


def load_text_data(
    dataset: str = "shakespeare",
    seq_len: int = 30,
    batch_size: int = 64,
    min_freq: int = 2,
    max_vocab: int = 10_000,
    data_dir: str = "data",
) -> Tuple[DataLoader, DataLoader, DataLoader, Vocabulary]:
    """
    Main entry point for Task 1 data loading.

    Args:
        dataset  : "shakespeare" | "wikitext2"
        seq_len  : context window length
        batch_size: mini-batch size
        min_freq : minimum token frequency for inclusion in vocabulary
        max_vocab : maximum vocabulary size
        data_dir : directory to cache downloaded files

    Returns:
        train_loader, val_loader, test_loader, vocab
    """

    # 1. Raw text
    if dataset == "wikitext2":
        train_text, val_text, test_text = _fetch_wikitext2(
            os.path.join(data_dir, "wikitext2")
        )
        if train_text is None:
            print("Falling back to Shakespeare dataset.")
            dataset = "shakespeare"

    if dataset == "shakespeare":
        # 80 / 10 / 10 split of the Shakespeare excerpt
        text = SHAKESPEARE_TEXT
        n = len(text)
        train_text = text[: int(0.8 * n)]
        val_text   = text[int(0.8 * n): int(0.9 * n)]
        test_text  = text[int(0.9 * n):]

    # 2. Tokenise
    train_tokens = simple_tokenise(train_text)
    val_tokens   = simple_tokenise(val_text)
    test_tokens  = simple_tokenise(test_text)

    # 3. Vocabulary (built on training data only)
    vocab = Vocabulary(min_freq=min_freq, max_size=max_vocab)
    vocab.build(train_tokens)
    print(f"  Vocabulary size: {len(vocab):,}")

    # 4. Encode
    train_ids = vocab.encode(train_tokens)
    val_ids   = vocab.encode(val_tokens)
    test_ids  = vocab.encode(test_tokens)

    # 5. Datasets & loaders
    train_ds = TextDataset(train_ids, seq_len)
    val_ds   = TextDataset(val_ids,   seq_len)
    test_ds  = TextDataset(test_ids,  seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

    print(
        f"  Splits — train: {len(train_ds):,}  val: {len(val_ds):,}  test: {len(test_ds):,} sequences"
    )
    return train_loader, val_loader, test_loader, vocab
