"""
task1_text_generation/train.py

Training and evaluation loop for Task 1 (Text Generation / Language Modelling).

Usage examples
──────────────
# Train LSTM with GloVe embeddings on Shakespeare
python task1_text_generation/train.py --model lstm --embedding glove --dataset shakespeare

# Train GRU with one-hot embeddings on WikiText-2
python task1_text_generation/train.py --model gru --embedding onehot --dataset wikitext2

# Run full comparison (all model × embedding combinations)
python task1_text_generation/train.py --run_all
"""

import os
import sys
import time
import math
import argparse
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from task1_text_generation.data_loader import load_text_data, Vocabulary
from task1_text_generation.models import LSTMLanguageModel, GRULanguageModel, RNNLanguageModel
from embeddings.embedding_utils import (
    build_glove_embedding,
    build_onehot_embedding,
    load_glove_vectors,
    GLOVE_DIM,
)
from utils.helpers import (
    set_seed,
    get_device,
    count_parameters,
    epoch_time,
    plot_training_curves,
    plot_comparison_bar,
    save_results,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIR = "results/task1"
os.makedirs(RESULTS_DIR, exist_ok=True)

DEFAULT_CONFIG = dict(
    dataset    = "shakespeare",  # "shakespeare" | "wikitext2"
    seq_len    = 30,
    batch_size = 64,
    embed_dim  = 256,          # used for trainable / one-hot projection
    hidden_size= 256,
    num_layers = 2,
    dropout    = 0.3,
    lr         = 1e-3,
    epochs     = 15,
    clip_grad  = 1.0,
    max_vocab  = 5_000,        # keep vocab small for one-hot feasibility
    seed       = 42,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_model(model_name: str, vocab: Vocabulary, embed_type: str,
                cfg: dict, glove_vectors=None, device=None):
    """Instantiate model with the requested embedding type."""

    vocab_size = len(vocab)
    vocab_dict = vocab._word2idx

    if embed_type == "glove":
        embedding = build_glove_embedding(vocab_dict, glove_vectors).to(device)
        embed_dim = GLOVE_DIM
    elif embed_type == "onehot":
        embedding = build_onehot_embedding(vocab_size, freeze=True).to(device)
        embed_dim = vocab_size          # one-hot dim = vocab_size
    else:
        embedding = None
        embed_dim = cfg["embed_dim"]

    model_cls = {"lstm": LSTMLanguageModel,
                 "gru":  GRULanguageModel,
                 "rnn":  RNNLanguageModel}[model_name]

    model = model_cls(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        embedding=embedding,
    ).to(device)

    print(f"  {model.model_name} + {embed_type}: {count_parameters(model):,} trainable params")
    return model


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, clip, device):
    model.train()
    total_loss = 0.0
    

    for inputs, targets in loader:
        inputs  = inputs.to(device)
        targets = targets.to(device)

        hidden = None

        optimizer.zero_grad()
        logits, hidden = model(inputs, hidden)
        # logits: (B, T, V)  targets: (B, T)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    

    with torch.no_grad():
        for inputs, targets in loader:
            inputs  = inputs.to(device)
            targets = targets.to(device)
            hidden = None
            logits, hidden = model(inputs, hidden)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            total_loss += loss.item()

    return total_loss / max(len(loader), 1)


# ---------------------------------------------------------------------------
# Text generation (greedy / temperature sampling)
# ---------------------------------------------------------------------------

def generate_text(
    model,
    vocab: Vocabulary,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    device=None,
) -> str:
    """Generate text autoregressively from a seed prompt."""
    model.eval()
    from task1_text_generation.data_loader import simple_tokenise

    tokens = simple_tokenise(prompt)
    ids    = vocab.encode(tokens)
    if not ids:
        ids = [vocab.unk_idx]

    src    = torch.tensor([ids], dtype=torch.long, device=device)
    hidden = None
    generated = list(tokens)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, hidden = model(src, hidden)
            last_logits = logits[0, -1] / temperature        # (V,)
            probs       = torch.softmax(last_logits, dim=-1)
            next_id     = torch.multinomial(probs, 1).item()
            next_word   = vocab.idx2word(next_id)
            generated.append(next_word)
            src = torch.tensor([[next_id]], dtype=torch.long, device=device)

    return " ".join(generated)


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------

def run_experiment(
    model_name: str,
    embed_type: str,
    cfg: dict,
    train_loader,
    val_loader,
    test_loader,
    vocab: Vocabulary,
    glove_vectors=None,
    device=None,
):
    tag = f"{model_name.upper()}_{embed_type}"
    print(f"\n{'='*60}")
    print(f"  Experiment: {tag}")
    print(f"{'='*60}")

    model     = build_model(model_name, vocab, embed_type, cfg, glove_vectors, device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["lr"])
    scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_path = os.path.join(RESULTS_DIR, f"best_{tag}.pt")

    for epoch in range(1, cfg["epochs"] + 1):
        start = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer,
                                 cfg["clip_grad"], device)
        val_loss   = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        mins, secs = epoch_time(start, time.time())
        train_ppl  = math.exp(train_loss)
        val_ppl    = math.exp(val_loss)
        print(
            f"  Epoch {epoch:02d}/{cfg['epochs']} | {mins}m{secs}s | "
            f"Train loss {train_loss:.3f} (PPL {train_ppl:.1f}) | "
            f"Val loss {val_loss:.3f} (PPL {val_ppl:.1f})"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)

    # --- test evaluation ---
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss = evaluate(model, test_loader, criterion, device)
    test_ppl  = math.exp(test_loss)
    print(f"\n  Test loss: {test_loss:.3f}  |  Test Perplexity: {test_ppl:.2f}")

    # --- generate sample text ---
    prompt = "to be or not to be"
    generated = generate_text(model, vocab, prompt, max_new_tokens=40, device=device)
    print(f"\n  Generated text (prompt: '{prompt}'):")
    print(f"  {generated}\n")

    # --- plots ---
    plot_training_curves(
        train_losses, val_losses,
        title=f"Training Curves — {tag}",
        save_path=os.path.join(RESULTS_DIR, f"curves_{tag}.png"),
    )

    return {
        "model": model_name.upper(),
        "embedding": embed_type,
        "test_loss": round(test_loss, 4),
        "test_ppl": round(test_ppl, 2),
        "generated": generated,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Task 1 — Text Generation")
    parser.add_argument("--model",     default="lstm",       choices=["lstm", "gru", "rnn"])
    parser.add_argument("--embedding", default="glove",      choices=["glove", "onehot", "trainable"])
    parser.add_argument("--dataset",   default="shakespeare", choices=["shakespeare", "wikitext2"])
    parser.add_argument("--epochs",    type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--run_all",   action="store_true",
                        help="Run all model × embedding combinations")
    args = parser.parse_args()

    cfg = {**DEFAULT_CONFIG, "epochs": args.epochs, "dataset": args.dataset}

    set_seed(cfg["seed"])
    device = get_device()
    print(f"Device: {device}")

    # Data
    print("\nLoading dataset …")
    train_loader, val_loader, test_loader, vocab = load_text_data(
        dataset   = cfg["dataset"],
        seq_len   = cfg["seq_len"],
        batch_size= cfg["batch_size"],
        max_vocab = cfg["max_vocab"],
    )

    # Pre-load GloVe once (reused across runs)
    glove_vectors = None
    try:
        from embeddings.embedding_utils import load_glove_vectors
        glove_vectors = load_glove_vectors()
    except Exception as e:
        print(f"  Warning: could not load GloVe ({e}). Runs with glove embedding will use random init.")

    # Experiments
    if args.run_all:
        combos = [
            ("lstm", "glove"),
            ("lstm", "onehot"),
            ("gru",  "glove"),
            ("gru",  "onehot"),
            ("rnn",  "glove"),
        ]
    else:
        combos = [(args.model, args.embedding)]

    all_results = []
    for model_name, embed_type in combos:
        result = run_experiment(
            model_name, embed_type, cfg,
            train_loader, val_loader, test_loader,
            vocab, glove_vectors, device,
        )
        all_results.append(result)

    # Summary table
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<10} {'Embedding':<12} {'Test Loss':>10} {'Test PPL':>10}")
    print("  " + "-" * 46)
    ppl_dict = {}
    for r in all_results:
        key = f"{r['model']} + {r['embedding']}"
        ppl_dict[key] = r["test_ppl"]
        print(f"  {r['model']:<10} {r['embedding']:<12} {r['test_loss']:>10.4f} {r['test_ppl']:>10.2f}")

    if len(ppl_dict) > 1:
        plot_comparison_bar(
            ppl_dict,
            metric="Test Perplexity",
            title="Task 1 — Model × Embedding Comparison",
            save_path=os.path.join(RESULTS_DIR, "comparison_ppl.png"),
        )

    save_results(
        {f"{r['model']}_{r['embedding']}_ppl": r["test_ppl"] for r in all_results},
        os.path.join(RESULTS_DIR, "results_summary.txt"),
    )


if __name__ == "__main__":
    main()
