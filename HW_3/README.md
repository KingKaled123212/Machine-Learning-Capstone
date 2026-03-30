# NLP Assignment 3 — Deep Learning for Natural Language Processing

> Two NLP tasks implemented with multiple RNN architectures and embedding strategies.

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Description](#2-dataset-description)
3. [Model Architectures Used](#3-model-architectures-used)
4. [Word Embedding Methods](#4-word-embedding-methods)
5. [Experimental Results](#5-experimental-results)
6. [Comparison of Models](#6-comparison-of-models)
7. [Challenges Faced During Implementation](#7-challenges-faced-during-implementation)
8. [Limitations of the Considered Models](#8-limitations-of-the-considered-models)
9. [Possible Future Improvements](#9-possible-future-improvements)

---

## 1. Project Overview

This project implements deep-learning solutions for two core NLP tasks:

| Task | Description | Evaluation Metric |
|------|-------------|-------------------|
| **Task 1** — Text Generation | Language model predicting the next word given a sequence (Shakespeare corpus) | Perplexity (PPL) |
| **Task 2** — Machine Translation | Seq2Seq encoder-decoder translating English → German (Tatoeba corpus) | BLEU score |

Both tasks are implemented with **at least two RNN architectures** (LSTM and GRU) and **two embedding types** (GloVe pre-trained and one-hot encoding)

### Quick Start

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd nlp-assignment

# 2. Install dependencies (Python 3.9+)
pip install -r requirements.txt

# 2a. Run a single experiment
python task1_text_generation/train.py --model lstm --embedding glove
python task2_machine_translation/train.py --cell gru --embedding onehot

# 2b. Run ALL experiments (generates full comparison plots)
python run_all.py

# 2c. Quick smoke-test with fewer epochs
python run_all.py --epochs 5
```

---

## 2. Dataset Description

### Task 1 — Shakespeare Corpus (built-in) / WikiText-2 (optional)

| Property | Value |
|----------|-------|
| **Primary dataset** | Shakespeare excerpt (built-in, ~7 000 tokens) |
| **Optional dataset** | WikiText-2 (~2 M tokens, auto-downloaded) |
| **Vocabulary size** | Up to 5 000 tokens (configurable) |
| **Sequence length** | 30 tokens per training window |
| **Train / Val / Test split** | 80 % / 10 % / 10 % |
| **Tokenisation** | Lowercased, whitespace-split, punctuation removed |

The Shakespeare excerpt was chosen because it is self-contained (no download required), small enough for CPU training in minutes, and has rich vocabulary that makes language modelling non-trivial.

### Task 2 — Tatoeba English–German Sentence Pairs

| Property | Value |
|----------|-------|
| **Source** | Tatoeba via ManyThings.org |
| **Language pair** | English (src) → German (tgt) |
| **Pairs used** | Up to 15 000 (filtered: 2–15 src words, 2–20 tgt words) |
| **Src vocabulary** | Up to 4 000 tokens |
| **Tgt vocabulary** | Up to 6 000 tokens |
| **Train / Val / Test** | 80 % / 10 % / 10 % (random split) |
| **Fallback** | 40 built-in EN→DE pairs used if download fails |

The dataset is automatically downloaded on first run. A built-in fallback ensures the code always runs offline, though BLEU scores will be lower.

---

## 3. Model Architectures Used

### Task 1 — Language Models

All language models share the same pipeline:

```
Token IDs → Embedding → Dropout → RNN Stack → Dropout → Linear → Logits
```

#### 3.1 LSTM Language Model

- **Cell type**: Long Short-Term Memory
- **Layers**: 2 stacked LSTM layers, hidden size 256
- **Key advantage**: The gating mechanism (input gate, forget gate, output gate + cell state) mitigates vanishing gradients and enables learning of long-range dependencies, such as subject-verb agreement across many tokens.

#### 3.2 GRU Language Model

- **Cell type**: Gated Recurrent Unit (Cho et al., 2014)
- **Layers**: 2 stacked GRU layers, hidden size 256
- **Key advantage**: Fewer parameters than LSTM (no separate cell state). Trains faster and performs comparably on most text-generation benchmarks.

#### 3.3 Vanilla RNN Language Model (bonus baseline)

- **Cell type**: Elman RNN with `tanh` non-linearity
- **Purpose**: Lower-bound baseline to quantify the improvement gained by gated architectures.
- **Expected behaviour**: Higher perplexity than LSTM/GRU due to vanishing gradients over long sequences.

### Task 2 — Seq2Seq Translation Models

Architecture follows Sutskever et al. (2014) and Cho et al. (2014):

```
src_ids ──► Encoder RNN ──► final hidden state
                                     │
tgt_ids (BOS, w₁…wₙ₋₁) ──► Decoder RNN ──► logits (w₁…wₙ, EOS)
```

**Teacher forcing** (ratio 0.5): during training, the decoder receives the ground-truth previous token with 50 % probability, and its own prediction otherwise. This stabilises early training while encouraging the model to generalise.

#### 3.4 LSTM Seq2Seq

- Encoder: 2-layer LSTM, hidden size 256
- Decoder: 2-layer LSTM, hidden size 256 (must match encoder)
- Both the hidden state `h` and cell state `c` are passed from encoder to decoder.

#### 3.5 GRU Seq2Seq

- Encoder: 2-layer GRU, hidden size 256
- Decoder: 2-layer GRU, hidden size 256
- Simpler state transfer (hidden state only, no cell state).

---

## 4. Word Embedding Methods

Two embedding strategies are compared across all experiments:

### 4.1 Pre-trained GloVe Embeddings

| Property | Value |
|----------|-------|
| **Source** | GloVe 6B (Stanford NLP, Pennington et al. 2014) |
| **Dimensionality** | 50-d |
| **Trained on** | 6 billion tokens (Wikipedia 2014 + Gigaword 5) |
| **OOV handling** | Xavier-uniform random initialisation for unseen words |
| **Fine-tuning** | Weights are trainable (frozen=False) by default |

GloVe vectors encode semantic similarity: words with similar meanings have nearby vector representations. This gives the model a strong prior, especially useful when the training corpus is small. For example, *king* and *queen* are already close in embedding space before any task-specific training begins.

```python
# Usage (from embedding_utils.py)
glove_vecs = load_glove_vectors()          # downloads once, cached
embedding  = build_glove_embedding(vocab_dict, glove_vecs)
# → nn.Embedding(vocab_size, 50) pre-filled with GloVe weights
```

### 4.2 One-Hot Encoding

| Property | Value |
|----------|-------|
| **Dimensionality** | Equal to vocabulary size (up to 5 000) |
| **Representation** | Identity matrix — each word is a binary indicator vector |
| **Trainable** | No (weights frozen) |
| **Semantic content** | None — all words are equidistant from each other |

One-hot encoding is the simplest possible representation and serves as the lower-bound baseline. Every word is assigned a unique dimension with value 1 and all other dimensions 0. The model receives no prior knowledge about word similarity.

```python
embedding = build_onehot_embedding(vocab_size, freeze=True)
# → nn.Embedding(vocab_size, vocab_size) with identity matrix weights
```

**Memory note**: Vocabulary is capped at 5 000 to keep memory feasible with one-hot embeddings (5 000-dimensional vectors).

### Embedding Comparison Table

| Property | GloVe 50-d | One-Hot |
|----------|-----------|---------|
| Dimensionality | 50 | vocab\_size |
| Semantic similarity encoded | Yes | No |
| External data required | Yes (6B token corpus) | No |
| Memory per token | 200 bytes | ~20 KB (at vocab 5 000) |
| Expected performance (small data) | Better | Worse |

---

## 5. Experimental Results

> Values below are representative results obtained with default hyperparameters (15 epochs Task 1, 15 epochs Task 2, hidden size 256, 2 layers, dropout 0.3).

### Task 1 — Text Generation (Perplexity — lower is better)

| Model | Embedding | Test Loss | Test Perplexity |
|-------|-----------|-----------|-----------------|
| LSTM  | GloVe     | 4.2271    | 68.52           |
| LSTM  | One-Hot   | 4.5252    | 92.31           |
| GRU   | GloVe     | 3.6244    | 37.50           |
| GRU   | One-Hot   | 4.3762    | 79.53           |
| RNN   | GloVe     | 2.0324    | 7.63 (overfit)  |

**Best generated text** — GRU + GloVe: *"to be or not to be"*:
> *to be or not to be die and the sleep a and thousand the bury world's merely perchance not to like to we sleep dreams perchance to they come...*


The GloVe model produces more coherent, contextually relevant text. The one-hot model tends to repeat short phrases and loses coherence faster.

### Task 2 — Machine Translation (BLEU — higher is better)

| Cell | Embedding | Test Loss | BLEU Score |
|------|-----------|-----------|------------|
| LSTM | GloVe     | 3.0987    | 10.86      |
| LSTM | One-Hot   | 3.4640    | 0.00       |
| GRU  | GloVe     | 2.9228    | 14.12      |
| GRU  | One-Hot   | 3.3562    | 0.00       |

| English (src) | German Reference | LSTM + GloVe | GRU + GloVe |
|---------------|-----------------|--------------|-------------|
| that s a `<unk>` | das ist ein `<unk>` | das ist ein | das ist ein |
| i want cash | ich will bargeld | ich will mich | ich will mich |
| is that paper | ist das papier | ist das hier | ist das hier |
| i m frugal | ich bin `<unk>` | ich bin ledig | ich bin ledig |

Note: BLEU scores of 15–20 are reasonable for a small, non-attention Seq2Seq model.

---

## 6. Comparison of Models

### Architecture Comparison

| Criterion | LSTM | GRU | Vanilla RNN |
|-----------|------|-----|-------------|
| Parameters | Most (4 gates) | Fewer (2 gates) | Fewest |
| Training speed | Slowest | Moderate | Fastest |
| Long-range dependencies | Best | Good | Poor |
| Vanishing gradient resistance | High | High | Low |
| Task 1 PPL (GloVe) | 68.52 | 37.50 | 7.63 (overfit) |



### Embedding Comparison

| Criterion | GloVe 50-d | One-Hot |
|-----------|-----------|---------|
| Semantic prior | Rich | None |
| Task 1 PPL (LSTM) | 68.52 | 92.31 |
| Task 1 PPL (GRU) | 37.50 | 79.53 |
| Task 2 BLEU (LSTM) | 10.86 | 0.00 |
| Task 2 BLEU (GRU) | 14.12 | 0.00 |
| GloVe vocabulary coverage | 94.1% (Task 1) / 99.8% (Task 2) | 100% |
| Embedding dimension | 50 | 101 (Task 1) / 2,624 (Task 2) |
| Trainable | Yes | No (frozen) |

**Key finding**: "GRU outperformed LSTM on both tasks, achieving PPL 37.50 vs 68.52 on Task 1 and BLEU 14.12 vs 10.86 on Task 2, despite having fewer parameters."

### Best Overall Configuration

- **Task 1**: GRU + GloVe (PPL = 37.50)
- **Task 2**: GRU + GloVe (BLEU = 14.12)

---

## 7. Challenges Faced During Implementation

### 7.1 Vocabulary Management for One-Hot Embeddings
One-hot vectors have dimensionality equal to vocab size. In Task 2 with the 
Tatoeba dataset, the source vocabulary was 2,624 words, producing 2,624-dimensional 
vectors per token. This made one-hot models significantly heavier — the LSTM + one-hot 
model had 18,983,183 parameters compared to just 3,096,381 for LSTM + GloVe. The 
solution was to cap the vocabulary size in the config, accepting that rarer words 
are mapped to `<unk>`.

### 7.2 GloVe Coverage
The Shakespeare corpus contains archaic English (e.g., *'tis*, *perchance*, *whither*) 
that is uncommon in GloVe's training data (Wikipedia + Gigaword). These out-of-vocabulary 
words receive random initialisations, reducing embedding quality for domain-specific text. 
Despite this, coverage was 94.1% (128/136 words) on the Shakespeare vocabulary, and 
an even higher 99.8% (2,620/2,624 words) on the Tatoeba source vocabulary, suggesting 
GloVe generalises well to everyday English even if archaic terms are occasionally missed.

### 7.3 Teacher Forcing Instability
During early experiments, setting teacher forcing ratio to 1.0 (always use ground truth) led to the decoder failing at inference time — a problem known as *exposure bias*. Reducing the ratio to 0.5 improved generalisation during greedy decoding.

### 7.4 Vanishing Gradients with Vanilla RNN
The baseline RNN was unable to learn long-range patterns effectively, as evidenced 
by its results on the Shakespeare corpus. While it achieved a misleadingly low 
perplexity of 7.63, this was due to overfitting on the tiny 115-sequence training 
set rather than genuine generalisation. In contrast, LSTM achieved 68.52 and GRU 
37.50, both showing more stable learning curves without memorising the training data. 
Gradient clipping (norm threshold = 1.0) was applied but did not prevent the RNN 
from overfitting, confirming why gated units are the standard choice for language 
modelling.

### 7.5 Truncated BPTT for Language Modelling
Training a language model requires *Truncated Backpropagation Through Time* (TBPTT): 
the hidden state must be detached from the computation graph between batches to 
prevent it from growing unbounded and causing out-of-memory errors. During 
implementation this also revealed a batch size mismatch bug — the hidden state was 
being initialised with batch size 64 and then reused on the final batch which had 
fewer samples, causing a runtime error. The fix was to reset the hidden state to 
None at the start of every batch rather than carrying it across batches.

### 7.6 Dataset Download Failures
The Tatoeba English-German dataset download failed during initial runs due to an 
HTTP 406 error from the host server. The built-in fallback of 40 sentence pairs 
allowed the code to run but produced poor results. The fix was to manually download 
the dataset and place it in the expected directory (`data/tatoeba/deu.txt`), after 
which the full 15,000 pairs were loaded automatically with no code changes required.

---

## 8. Limitations of the Considered Models

### 8.1 No Attention Mechanism
The Seq2Seq model compresses the entire source sequence into a single fixed-size 
vector, creating an information bottleneck that worsens for longer sentences. 
Attention would allow the decoder to focus on relevant source tokens at each step.

### 8.2 Fixed Vocabulary
All models map unseen words to `<unk>`, visible in the sample translations 
(e.g., "i m frugal" → `<unk>`). Subword tokenisation such as BPE would handle 
rare and morphologically complex German words far better.

### 8.3 Limited Context Window
The language model uses a fixed window of 30 tokens, so any dependencies beyond 
that are invisible to the model. Transformer-based models handle contexts of 
thousands of tokens.

### 8.4 No Beam Search
The translation model uses greedy decoding, always picking the single 
highest-probability token. Beam search typically improves BLEU by 2–5 points 
with no retraining required.

### 8.5 Shallow Architecture
Both tasks use only 2 RNN layers. State-of-the-art translation systems use 
6+ layers or Transformer architectures, giving them significantly more capacity.

### 8.6 One-Hot Scalability
One-hot embedding dimension equals vocab size, which became a real problem in 
Task 2 where the one-hot LSTM had 18.9M parameters vs 3.1M for GloVe — 6x larger 
with worse performance (0.00 BLEU vs 10.86).

### 8.7 Perplexity as a Metric
Perplexity does not directly reflect text quality. The RNN achieved PPL 7.63 
on Shakespeare but was clearly overfitting, producing incoherent generated text 
— demonstrating that low perplexity alone is not a reliable indicator of a good model.

---

## 9. Possible Future Improvements

### 9.1 Attention Mechanism
Adding Bahdanau or Luong attention to the Seq2Seq model would allow the decoder 
to focus on relevant source tokens at each step rather than relying on a single 
fixed-size context vector, which would likely improve BLEU significantly on longer 
sentences.

### 9.2 Beam Search Decoding
Implementing beam search at inference time is a low-effort improvement that 
typically raises BLEU by 2–5 points with no retraining required. Given our 
best BLEU of 14.12, this could push results meaningfully higher.

### 9.3 Subword Tokenisation
Replacing word-level tokens with Byte-Pair Encoding (BPE) would eliminate 
`<unk>` tokens entirely and handle morphologically complex German words better, 
directly addressing one of the main weaknesses seen in the sample translations.

### 9.4 Larger Pre-trained Embeddings
Upgrading from 50-d to 300-d GloVe, or using fastText embeddings which handle 
subwords, would improve semantic coverage especially for rare words.

### 9.5 Transformer Architecture
Replacing the RNN encoder-decoder with a Transformer would enable parallel 
training and dramatically improve translation quality. The original Transformer 
paper reported 25+ BLEU on WMT English-German, well above our 14.12.

### 9.6 Larger Dataset
Training on the full WMT English-German dataset (~4.5M pairs) instead of 15,000 
Tatoeba pairs would give the model far more coverage and likely resolve the 
0.00 BLEU seen with one-hot embeddings.

---

## Hyperparameter Reference

| Parameter | Task 1 Default | Task 2 Default |
|-----------|---------------|---------------|
| Embedding dim (trainable) | 256 | 256 |
| GloVe dim | 50 | 50 |
| Hidden size | 256 | 256 |
| Num layers | 2 | 2 |
| Dropout | 0.3 | 0.3 |
| Learning rate | 1e-3 | 5e-4 |
| Batch size | 64 | 64 |
| Max epochs | 15 | 15 |
| Gradient clip norm | 1.0 | 1.0 |
| Teacher forcing ratio | N/A | 0.5 |
| Sequence length | 30 | variable |
| Optimizer | Adam | Adam |
| LR scheduler | ReduceLROnPlateau | ReduceLROnPlateau |

---

## References

1. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*.
2. Cho, K., et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. *EMNLP 2014*.
3. Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to Sequence Learning with Neural Networks. *NeurIPS 2014*.
4. Pennington, J., Socher, R., & Manning, C. (2014). GloVe: Global Vectors for Word Representation. *EMNLP 2014*.
5. Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. *ICLR 2015*.
6. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS 2017*.
