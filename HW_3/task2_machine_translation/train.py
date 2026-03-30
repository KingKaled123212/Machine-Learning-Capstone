"""
task2_machine_translation/train.py

Training and evaluation for Task 2 (Machine Translation).

Usage
─────
# Train LSTM Seq2Seq with GloVe source embeddings
python task2_machine_translation/train.py --cell lstm --embedding glove

# Train GRU with one-hot embeddings
python task2_machine_translation/train.py --cell gru --embedding onehot

# Run all combinations
python task2_machine_translation/train.py --run_all
"""

import os
import sys
import time
import math
import argparse
from typing import List, Dict

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from task2_machine_translation.data_loader import load_translation_data, TranslationVocabulary
from task2_machine_translation.models import build_seq2seq
from embeddings.embedding_utils import (
    build_glove_embedding,
    build_onehot_embedding,
    load_glove_vectors,
    GLOVE_DIM,
)
from utils.helpers import (
    set_seed, get_device, count_parameters, epoch_time,
    plot_training_curves, plot_comparison_bar, save_results,
)

RESULTS_DIR = "results/task2"
os.makedirs(RESULTS_DIR, exist_ok=True)

DEFAULT_CONFIG = dict(
    embed_dim    = 256,
    hidden_size  = 256,
    num_layers   = 2,
    dropout      = 0.3,
    lr           = 5e-4,
    epochs       = 20,
    clip_grad    = 1.0,
    batch_size   = 64,
    tf_ratio     = 0.5,    # teacher forcing ratio
    max_vocab_src= 4_000,
    max_vocab_tgt= 6_000,
    max_pairs    = 15_000,
    seed         = 42,
)


# ---------------------------------------------------------------------------
# BLEU score (simple corpus-level implementation)
# ---------------------------------------------------------------------------

def compute_bleu(
    hypotheses: List[List[str]],
    references: List[List[str]],
    max_n: int = 4,
) -> float:
    """
    Corpus-level BLEU score (simplified, without brevity penalty smoothing).
    Uses sacrebleu if available, else falls back to a simple n-gram implementation.
    """
    try:
        import sacrebleu
        # sacrebleu expects string references
        hyp_strs = [" ".join(h) for h in hypotheses]
        ref_strs = [" ".join(r) for r in references]
        bleu = sacrebleu.corpus_bleu(hyp_strs, [ref_strs])
        return bleu.score
    except ImportError:
        pass

    from collections import Counter
    import math

    def get_ngrams(tokens, n):
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    clipped_matches = [0] * max_n
    total_hyp_ngrams = [0] * max_n

    hyp_len = sum(len(h) for h in hypotheses)
    ref_len = sum(len(r) for r in references)

    for hyp, ref in zip(hypotheses, references):
        for n in range(1, max_n + 1):
            hyp_ngrams = get_ngrams(hyp, n)
            ref_ngrams = get_ngrams(ref, n)
            matches = {k: min(v, ref_ngrams[k]) for k, v in hyp_ngrams.items()}
            clipped_matches[n-1] += sum(matches.values())
            total_hyp_ngrams[n-1] += max(len(hyp) - n + 1, 0)

    log_bleu = 0.0
    for n in range(max_n):
        if clipped_matches[n] == 0 or total_hyp_ngrams[n] == 0:
            log_bleu += float("-inf")
            break
        log_bleu += math.log(clipped_matches[n] / total_hyp_ngrams[n])

    bp = min(1.0, math.exp(1 - ref_len / max(hyp_len, 1)))
    bleu = bp * math.exp(log_bleu / max_n)
    return bleu * 100.0


# ---------------------------------------------------------------------------
# Train / evaluate helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, clip, device, tgt_pad_idx):
    model.train()
    total_loss = 0.0

    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)

        optimizer.zero_grad()
        # logits: (B, T-1, V), tgt input: tgt[:,:-1], target: tgt[:,1:]
        logits = model(src, tgt)
        tgt_len = logits.size(1)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt[:, 1:tgt_len+1].reshape(-1),
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)
        logits = model(src, tgt)
        tgt_len = logits.size(1)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt[:, 1:tgt_len+1].reshape(-1),
        )
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate_bleu(model, loader, tgt_vocab: TranslationVocabulary, device) -> float:
    model.eval()
    hypotheses, references = [], []

    for src, tgt in loader:
        src = src.to(device)
        preds = model.translate(src, max_len=50, device=device)  # (B, L)

        for i in range(src.size(0)):
            hyp_ids = preds[i].tolist()
            ref_ids = tgt[i].tolist()
            hypotheses.append(tgt_vocab.decode(hyp_ids, skip_special=True).split())
            references.append(tgt_vocab.decode(ref_ids, skip_special=True).split())

    return compute_bleu(hypotheses, references)


# ---------------------------------------------------------------------------
# Sample translations
# ---------------------------------------------------------------------------

def show_translations(model, test_loader, src_vocab, tgt_vocab, device, n=5):
    model.eval()
    count = 0
    print("\n  Sample translations:")
    print("  " + "-" * 60)
    with torch.no_grad():
        for src, tgt in test_loader:
            src = src.to(device)
            preds = model.translate(src, max_len=50)
            for i in range(min(n - count, src.size(0))):
                src_text = src_vocab.decode(src[i].tolist())
                ref_text = tgt_vocab.decode(tgt[i].tolist())
                hyp_text = tgt_vocab.decode(preds[i].tolist())
                print(f"  SRC : {src_text}")
                print(f"  REF : {ref_text}")
                print(f"  HYP : {hyp_text}")
                print()
                count += 1
                if count >= n:
                    return


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def run_experiment(
    cell_type: str,
    embed_type: str,
    cfg: dict,
    train_loader, val_loader, test_loader,
    src_vocab: TranslationVocabulary,
    tgt_vocab: TranslationVocabulary,
    glove_vectors=None,
    device=None,
):
    tag = f"{cell_type.upper()}_{embed_type}"
    print(f"\n{'='*60}")
    print(f"  Experiment: {tag}")
    print(f"{'='*60}")

    # Build source embedding
    if embed_type == "glove":
        src_emb = build_glove_embedding(src_vocab.w2i, glove_vectors).to(device)
    elif embed_type == "onehot":
        src_emb = build_onehot_embedding(len(src_vocab), freeze=True).to(device)
    else:
        src_emb = None

    # Resolve embed_dim
    if src_emb is not None:
        embed_dim = src_emb.embedding_dim
    else:
        embed_dim = cfg["embed_dim"]

    model = build_seq2seq(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        tgt_bos_idx=tgt_vocab.BOS_IDX,
        tgt_eos_idx=tgt_vocab.EOS_IDX,
        embed_dim=embed_dim,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        cell_type=cell_type.upper(),
        src_embedding=src_emb,
        teacher_forcing_ratio=cfg["tf_ratio"],
    ).to(device)

    print(f"  Params: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.PAD_IDX)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["lr"])
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_path = os.path.join(RESULTS_DIR, f"best_{tag}.pt")

    for epoch in range(1, cfg["epochs"] + 1):
        start = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer,
                                  cfg["clip_grad"], device, tgt_vocab.PAD_IDX)
        val_loss   = evaluate_loss(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        mins, secs = epoch_time(start, time.time())
        print(f"  Epoch {epoch:02d}/{cfg['epochs']} | {mins}m{secs}s | "
              f"Train {train_loss:.4f} | Val {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)

    # Load best model
    model.load_state_dict(torch.load(best_path, map_location=device))

    test_loss = evaluate_loss(model, test_loader, criterion, device)
    bleu      = evaluate_bleu(model, test_loader, tgt_vocab, device)
    print(f"\n  Test Loss: {test_loss:.4f}  |  BLEU: {bleu:.2f}")

    show_translations(model, test_loader, src_vocab, tgt_vocab, device, n=4)

    plot_training_curves(
        train_losses, val_losses,
        title=f"MT Training Curves — {tag}",
        save_path=os.path.join(RESULTS_DIR, f"curves_{tag}.png"),
    )

    return {
        "cell": cell_type.upper(),
        "embedding": embed_type,
        "test_loss": round(test_loss, 4),
        "bleu": round(bleu, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Task 2 — Machine Translation")
    parser.add_argument("--cell",      default="lstm", choices=["lstm", "gru"])
    parser.add_argument("--embedding", default="glove", choices=["glove", "onehot", "trainable"])
    parser.add_argument("--epochs",    type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--run_all",   action="store_true")
    args = parser.parse_args()

    cfg = {**DEFAULT_CONFIG, "epochs": args.epochs}

    set_seed(cfg["seed"])
    device = get_device()
    print(f"Device: {device}")

    print("\nLoading translation dataset …")
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = load_translation_data(
        use_tatoeba  = True,
        max_pairs    = cfg["max_pairs"],
        batch_size   = cfg["batch_size"],
        src_vocab_size = cfg["max_vocab_src"],
        tgt_vocab_size = cfg["max_vocab_tgt"],
    )

    # Pre-load GloVe
    glove_vectors = None
    try:
        glove_vectors = load_glove_vectors()
    except Exception as e:
        print(f"  Warning: GloVe load failed ({e})")

    combos = (
        [("lstm", "glove"), ("lstm", "onehot"), ("gru", "glove"), ("gru", "onehot")]
        if args.run_all
        else [(args.cell, args.embedding)]
    )

    all_results = []
    for cell_type, embed_type in combos:
        r = run_experiment(
            cell_type, embed_type, cfg,
            train_loader, val_loader, test_loader,
            src_vocab, tgt_vocab,
            glove_vectors, device,
        )
        all_results.append(r)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'Cell':<8} {'Embedding':<12} {'Test Loss':>10} {'BLEU':>8}")
    print("  " + "-" * 42)
    bleu_dict = {}
    for r in all_results:
        key = f"{r['cell']} + {r['embedding']}"
        bleu_dict[key] = r["bleu"]
        print(f"  {r['cell']:<8} {r['embedding']:<12} {r['test_loss']:>10.4f} {r['bleu']:>8.2f}")

    if len(bleu_dict) > 1:
        plot_comparison_bar(
            bleu_dict,
            metric="BLEU Score",
            title="Task 2 — MT Model × Embedding Comparison",
            save_path=os.path.join(RESULTS_DIR, "comparison_bleu.png"),
        )

    save_results(
        {f"{r['cell']}_{r['embedding']}_bleu": r["bleu"] for r in all_results},
        os.path.join(RESULTS_DIR, "results_summary.txt"),
    )


if __name__ == "__main__":
    main()
