"""
task2_machine_translation/data_loader.py

Loads a small English → German (or English → Spanish) parallel corpus for
machine translation.  Two sources are supported:

  1. Tatoeba sentence pairs  — small, easy to download (~200 k pairs)
  2. A tiny built-in corpus  — used as a fallback when network is unavailable

The module exposes:
  - TranslationVocabulary : per-language vocabulary with BOS/EOS/PAD/UNK tokens
  - TranslationDataset    : Dataset yielding (src_ids, tgt_ids) pairs
  - get_translation_loaders() : convenience wrapper
"""

import os
import re
import urllib.request
import zipfile
import random
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# ---------------------------------------------------------------------------
# Tiny built-in corpus (English → German)
# ---------------------------------------------------------------------------

BUILTIN_PAIRS = [
    ("hello", "hallo"),
    ("good morning", "guten morgen"),
    ("good night", "gute nacht"),
    ("thank you", "danke schoen"),
    ("how are you", "wie geht es ihnen"),
    ("what is your name", "wie heissen sie"),
    ("my name is john", "mein name ist john"),
    ("i love you", "ich liebe dich"),
    ("where is the hotel", "wo ist das hotel"),
    ("i am hungry", "ich habe hunger"),
    ("the cat is on the mat", "die katze sitzt auf der matte"),
    ("please help me", "bitte helfen sie mir"),
    ("i do not understand", "ich verstehe nicht"),
    ("the weather is nice today", "das wetter ist heute schoen"),
    ("i want to eat pizza", "ich moechte pizza essen"),
    ("can you speak english", "koennen sie englisch sprechen"),
    ("how much does this cost", "was kostet das"),
    ("i am from the united states", "ich komme aus den vereinigten staaten"),
    ("the book is on the table", "das buch liegt auf dem tisch"),
    ("she is very beautiful", "sie ist sehr schoen"),
    ("we are going to the park", "wir gehen in den park"),
    ("i need a doctor", "ich brauche einen arzt"),
    ("call the police", "rufen sie die polizei"),
    ("i am tired", "ich bin muede"),
    ("the train is late", "der zug hat verspaetung"),
    ("do you have a map", "haben sie eine karte"),
    ("i like music", "ich mag musik"),
    ("she reads a book", "sie liest ein buch"),
    ("he plays football", "er spielt fussball"),
    ("we eat dinner together", "wir essen gemeinsam abendessen"),
    ("the children are playing", "die kinder spielen"),
    ("it is raining outside", "es regnet draussen"),
    ("i will see you tomorrow", "ich werde dich morgen sehen"),
    ("the sun is shining", "die sonne scheint"),
    ("i enjoy reading books", "ich lese gerne buecher"),
    ("he works at the hospital", "er arbeitet im krankenhaus"),
    ("she studies mathematics", "sie studiert mathematik"),
    ("the dog is barking", "der hund bellt"),
    ("we traveled to berlin", "wir sind nach berlin gereist"),
    ("i speak a little german", "ich spreche ein bisschen deutsch"),
] * 50   # replicate to create a small but workable training set


# ---------------------------------------------------------------------------
# Download Tatoeba pairs
# ---------------------------------------------------------------------------

TATOEBA_URL = (
    "https://www.manythings.org/anki/deu-eng.zip"  # English ↔ German
)


def _download_tatoeba(data_dir: str = "data/tatoeba") -> Optional[str]:
    """
    Download the Tatoeba English-German sentence pairs.
    Returns path to the extracted .txt file, or None on failure.
    """
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "deu-eng.zip")
    txt_path = os.path.join(data_dir, "deu.txt")

    if os.path.exists(txt_path):
        return txt_path

    print(f"  Downloading Tatoeba English-German pairs …")
    try:
        urllib.request.urlretrieve(TATOEBA_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(data_dir)
        # The extracted file is called "deu.txt"
        return txt_path
    except Exception as e:
        print(f"  Download failed: {e}")
        return None


def _parse_tatoeba(path: str, max_pairs: int = 20_000) -> List[Tuple[str, str]]:
    """Parse Tatoeba tab-separated file into (english, german) pairs."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                en, de = parts[0].strip(), parts[1].strip()
                # Simple length filter
                if 2 <= len(en.split()) <= 15 and 2 <= len(de.split()) <= 20:
                    pairs.append((en, de))
            if len(pairs) >= max_pairs:
                break
    return pairs


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class TranslationVocabulary:
    """Word-level vocabulary for one language in a translation pair."""

    PAD  = "<pad>"
    UNK  = "<unk>"
    BOS  = "<bos>"
    EOS  = "<eos>"
    SPECIALS = [PAD, UNK, BOS, EOS]

    PAD_IDX = 0
    UNK_IDX = 1
    BOS_IDX = 2
    EOS_IDX = 3

    def __init__(self, min_freq: int = 1, max_size: Optional[int] = None):
        self.min_freq = min_freq
        self.max_size = max_size
        self.w2i: Dict[str, int] = {}
        self.i2w: Dict[int, str] = {}

    def build(self, sentences: List[List[str]]) -> "TranslationVocabulary":
        from collections import Counter
        freq: Counter = Counter(w for sent in sentences for w in sent)
        candidates = [w for w, c in freq.most_common() if c >= self.min_freq]
        if self.max_size:
            candidates = candidates[: self.max_size - len(self.SPECIALS)]
        for i, tok in enumerate(self.SPECIALS + candidates):
            self.w2i[tok] = i
            self.i2w[i]   = tok
        return self

    def __len__(self) -> int:
        return len(self.w2i)

    def encode(self, tokens: List[str], add_bos: bool = True, add_eos: bool = True) -> List[int]:
        ids = [self.w2i.get(t, self.UNK_IDX) for t in tokens]
        if add_bos:
            ids = [self.BOS_IDX] + ids
        if add_eos:
            ids = ids + [self.EOS_IDX]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        skip = {self.PAD_IDX, self.BOS_IDX, self.EOS_IDX} if skip_special else set()
        return " ".join(self.i2w.get(i, self.UNK) for i in ids if i not in skip)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-zäöüßа-яA-Za-z0-9\s]", " ", text)
    return text.split()


class TranslationDataset(Dataset):
    """
    Parallel corpus dataset.

    Each item is (src_ids, tgt_ids) as LongTensors with BOS/EOS added.
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        src_vocab: TranslationVocabulary,
        tgt_vocab: TranslationVocabulary,
    ):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.data: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for src_text, tgt_text in pairs:
            src_ids = src_vocab.encode(_tokenise(src_text))
            tgt_ids = tgt_vocab.encode(_tokenise(tgt_text))
            self.data.append(
                (torch.tensor(src_ids, dtype=torch.long),
                 torch.tensor(tgt_ids, dtype=torch.long))
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


def _collate_fn(batch, src_pad, tgt_pad):
    """Pad a batch of variable-length sequences."""
    src_seqs, tgt_seqs = zip(*batch)
    src_padded = pad_sequence(src_seqs, batch_first=True, padding_value=src_pad)
    tgt_padded = pad_sequence(tgt_seqs, batch_first=True, padding_value=tgt_pad)
    return src_padded, tgt_padded


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_translation_data(
    use_tatoeba: bool = True,
    max_pairs: int = 10_000,
    batch_size: int = 64,
    src_vocab_size: int = 5_000,
    tgt_vocab_size: int = 7_000,
    data_dir: str = "data",
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader,
           TranslationVocabulary, TranslationVocabulary]:
    """
    Main entry point for Task 2 data loading.

    Returns:
        train_loader, val_loader, test_loader, src_vocab, tgt_vocab
    """
    random.seed(seed)

    # 1. Load raw pairs
    pairs = None
    if use_tatoeba:
        txt_path = _download_tatoeba(os.path.join(data_dir, "tatoeba"))
        if txt_path and os.path.exists(txt_path):
            pairs = _parse_tatoeba(txt_path, max_pairs=max_pairs)
            print(f"  Loaded {len(pairs):,} Tatoeba sentence pairs")

    if not pairs:
        print("  Using built-in English-German pairs (fallback)")
        pairs = BUILTIN_PAIRS

    # 2. Shuffle and split
    random.shuffle(pairs)
    n      = len(pairs)
    n_val  = max(1, int(0.1 * n))
    n_test = max(1, int(0.1 * n))
    train_pairs = pairs[: n - n_val - n_test]
    val_pairs   = pairs[n - n_val - n_test : n - n_test]
    test_pairs  = pairs[n - n_test :]

    print(f"  Train: {len(train_pairs):,}  Val: {len(val_pairs):,}  Test: {len(test_pairs):,}")

    # 3. Vocabularies (built on training data only)
    src_sents = [_tokenise(p[0]) for p in train_pairs]
    tgt_sents = [_tokenise(p[1]) for p in train_pairs]

    src_vocab = TranslationVocabulary(max_size=src_vocab_size).build(src_sents)
    tgt_vocab = TranslationVocabulary(max_size=tgt_vocab_size).build(tgt_sents)

    print(f"  Src vocab: {len(src_vocab):,}  Tgt vocab: {len(tgt_vocab):,}")

    # 4. Datasets
    train_ds = TranslationDataset(train_pairs, src_vocab, tgt_vocab)
    val_ds   = TranslationDataset(val_pairs,   src_vocab, tgt_vocab)
    test_ds  = TranslationDataset(test_pairs,  src_vocab, tgt_vocab)

    from functools import partial
    collate = partial(_collate_fn, src_pad=src_vocab.PAD_IDX, tgt_pad=tgt_vocab.PAD_IDX)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=False)

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab
