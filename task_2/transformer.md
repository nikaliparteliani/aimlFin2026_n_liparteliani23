# Transformer Networks and Their Applications in Cybersecurity

## 1. Introduction

The **Transformer** is a revolutionary deep learning architecture introduced in the landmark paper *"Attention Is All You Need"* (Vaswani et al., 2017). Unlike previous sequence models such as Recurrent Neural Networks (RNNs) and Long Short-Term Memory networks (LSTMs), Transformers do not rely on sequential processing. Instead, they use a mechanism called **Self-Attention** to process all elements of an input sequence simultaneously in parallel, capturing long-range dependencies far more efficiently.

Transformers have become the backbone of modern artificial intelligence, powering state-of-the-art models across natural language processing (BERT, GPT), computer vision (Vision Transformer), speech recognition, and increasingly, **cybersecurity** applications such as malware detection, intrusion detection, phishing analysis, and threat intelligence.

The key innovation of the Transformer lies in its ability to dynamically weigh the importance of different parts of the input when producing an output — this is the essence of the **attention mechanism**. Rather than compressing an entire sequence into a fixed-length vector (as RNNs do), Transformers maintain direct access to every position in the sequence, allowing them to model complex relationships regardless of distance.

---

## 2. Transformer Architecture Overview

The Transformer follows an **Encoder-Decoder** structure, though many modern applications use only the encoder (BERT) or only the decoder (GPT).

```
              COMPLETE TRANSFORMER ARCHITECTURE
    ════════════════════════════════════════════════════

                        ┌─────────────┐
                        │   OUTPUT    │
                        │ Probabilities│
                        └──────┬──────┘
                               │
                        ┌──────┴──────┐
                        │   Softmax   │
                        └──────┬──────┘
                               │
                        ┌──────┴──────┐
                        │   Linear    │
                        └──────┬──────┘
                               │
               ┌───────────────┴───────────────┐
               │          DECODER              │
               │  ┌─────────────────────────┐  │
               │  │  Feed Forward Network   │  │
               │  └────────────┬────────────┘  │
               │               │               │
               │  ┌────────────┴────────────┐  │
               │  │  Multi-Head Attention   │◄─┼──── Encoder Output
               │  │  (Cross-Attention)      │  │
               │  └────────────┬────────────┘  │
               │               │               │
               │  ┌────────────┴────────────┐  │
               │  │  Masked Multi-Head      │  │
               │  │  Self-Attention         │  │
               │  └────────────┬────────────┘  │
               │               │               │
               │  ┌────────────┴────────────┐  │
               │  │  Positional Encoding    │  │
               │  │  + Output Embedding     │  │
               │  └────────────┬────────────┘  │
               └───────────────┬───────────────┘
                               │
               ┌───────────────┴───────────────┐
               │          ENCODER              │
               │  ┌─────────────────────────┐  │
               │  │  Feed Forward Network   │  │
               │  └────────────┬────────────┘  │
               │               │               │
               │  ┌────────────┴────────────┐  │
               │  │  Multi-Head             │  │
               │  │  Self-Attention         │  │
               │  └────────────┬────────────┘  │
               │               │               │
               │  ┌────────────┴────────────┐  │
               │  │  Positional Encoding    │  │
               │  │  + Input Embedding      │  │
               │  └────────────┬────────────┘  │
               └───────────────┬───────────────┘
                               │
                        ┌──────┴──────┐
                        │   INPUT     │
                        │  Sequence   │
                        └─────────────┘
```

### 2.1 Core Components

The Transformer consists of these fundamental building blocks:

- **Input Embedding:** Converts tokens (words, bytes, network features) into dense vector representations
- **Positional Encoding:** Injects information about the position of each token in the sequence
- **Multi-Head Self-Attention:** The core mechanism that allows each token to attend to all other tokens
- **Feed-Forward Network:** A fully connected network applied independently to each position
- **Layer Normalization & Residual Connections:** Stabilize training and enable deeper networks
- **Output Linear Layer + Softmax:** Produces final probability distribution over possible outputs

---

## 3. The Self-Attention Mechanism (Detailed Visualization)

Self-Attention is the heart of the Transformer. It allows each element in a sequence to look at every other element and determine how much "attention" to pay to each one.

### 3.1 Query, Key, Value (Q, K, V)

For each input token, three vectors are computed:

- **Query (Q):** "What am I looking for?"
- **Key (K):** "What do I contain?"
- **Value (V):** "What information do I provide?"

```
                    Q, K, V COMPUTATION
    ════════════════════════════════════════════

    Input Embeddings          Weight Matrices         Q, K, V Vectors
    ┌─────────────┐          ┌──────┐
    │ Token 1: x₁ │───┬──────│  Wq  │──────→  q₁  (Query)
    │             │   │      └──────┘
    │             │   ├──────┌──────┐
    │             │   │      │  Wk  │──────→  k₁  (Key)
    │             │   │      └──────┘
    │             │   └──────┌──────┐
    │             │          │  Wv  │──────→  v₁  (Value)
    └─────────────┘          └──────┘

    ┌─────────────┐          (same W matrices)
    │ Token 2: x₂ │──────────────────────→  q₂, k₂, v₂
    └─────────────┘
    ┌─────────────┐
    │ Token 3: x₃ │──────────────────────→  q₃, k₃, v₃
    └─────────────┘
```

### 3.2 Attention Score Calculation

The attention formula:

```
                              Q × Kᵀ
    Attention(Q, K, V) = softmax( ─────── ) × V
                              √d_k
```

Where `d_k` is the dimension of the key vectors (scaling prevents large dot products).

### 3.3 Step-by-Step Attention Example

Consider the sentence: **"The firewall blocked the attack"**

```
    STEP 1: Compute Attention Scores (Q × Kᵀ / √d_k)
    ══════════════════════════════════════════════════════

              Key→   The    firewall  blocked   the    attack
    Query↓    ┌──────┬─────────┬────────┬──────┬────────┐
    The       │ 0.15 │  0.05   │  0.10  │ 0.60 │  0.10  │
    firewall  │ 0.08 │  0.30   │  0.35  │ 0.07 │  0.20  │
    blocked   │ 0.05 │  0.25   │  0.10  │ 0.10 │  0.50  │
    the       │ 0.55 │  0.10   │  0.05  │ 0.20 │  0.10  │
    attack    │ 0.05 │  0.20   │  0.45  │ 0.05 │  0.25  │
              └──────┴─────────┴────────┴──────┴────────┘

    (Values after softmax normalization — each row sums to 1.0)


    STEP 2: Visualize Attention Weights
    ══════════════════════════════════════════════════════

    "blocked" attends most strongly to "attack" (0.50)
    → The model learns that "blocked" is semantically connected to "attack"

     The ──────── firewall ──────── blocked ──────── the ──────── attack
      │              │            ╱    │    ╲          │           │
      │              │          ╱     │      ╲        │           │
      │              │        ╱      │        ╲       │           │
      │              │      ╱       │          ╲      │           │
      ▼              ▼    ▼        ▼            ▼    ▼           ▼
    [0.05]        [0.25] [0.10]  [0.10]       [0.50]
                                                  ▲
                                          Strongest attention:
                                       "blocked" → "attack"


    STEP 3: Compute Weighted Sum of Values
    ══════════════════════════════════════════════════════

    Output for "blocked" = 0.05×V(The) + 0.25×V(firewall)
                         + 0.10×V(blocked) + 0.10×V(the)
                         + 0.50×V(attack)

    → The new representation of "blocked" is enriched with
      context from all other words, especially "attack"
```

### 3.4 Multi-Head Attention

Instead of computing attention once, the Transformer uses **multiple attention heads** in parallel, each learning different types of relationships:

```
    MULTI-HEAD ATTENTION MECHANISM
    ════════════════════════════════════════════════════

    Input ──┬──────────┬──────────┬──────────┐
            │          │          │          │
            ▼          ▼          ▼          ▼
       ┌─────────┐┌─────────┐┌─────────┐┌─────────┐
       │ Head 1  ││ Head 2  ││ Head 3  ││ Head 4  │
       │Syntactic││Semantic ││Positional││ Entity  │
       │Relations││Similarity││ Patterns ││Relations│
       └────┬────┘└────┬────┘└────┬────┘└────┬────┘
            │          │          │          │
            └────┬─────┴────┬─────┴────┬─────┘
                 │          │          │
                 ▼          ▼          ▼
              ┌─────────────────────────────┐
              │       Concatenate           │
              └──────────────┬──────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │    Linear Projection (Wo)   │
              └──────────────┬──────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │   Multi-Head Output         │
              └─────────────────────────────┘

    Each head captures DIFFERENT patterns:
    • Head 1: "blocked" → "firewall" (what does the blocking)
    • Head 2: "blocked" → "attack"   (what was blocked)
    • Head 3: "the"     → "attack"   (article-noun pairing)
    • Head 4: "firewall"→ "blocked"  (subject-verb relation)
```

---

## 4. Positional Encoding (Detailed Visualization)

Since Transformers process all tokens in parallel (no recurrence), they have no inherent sense of word order. **Positional Encoding** solves this by adding position-dependent signals to the input embeddings.

### 4.1 Sinusoidal Positional Encoding Formula

```
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Where:
    • pos   = position of the token in the sequence (0, 1, 2, ...)
    • i     = dimension index
    • d_model = model dimension (e.g., 512)
```

### 4.2 How Positional Encoding Works

```
    POSITIONAL ENCODING VISUALIZATION
    ════════════════════════════════════════════════════

    Token Embeddings:          Positional Encodings:        Final Input:
    ┌──────────────┐          ┌──────────────┐            ┌──────────────┐
    │"The"  = [0.2,│          │PE(0) = [0.00,│            │[0.20, 1.20,  │
    │  1.5, -0.3,  │    +     │  0.30, 1.00, │     =      │  0.70, 0.54, │
    │  0.8, 0.1]   │          │ -0.46, 0.15] │            │  0.25]       │
    └──────────────┘          └──────────────┘            └──────────────┘

    ┌──────────────┐          ┌──────────────┐            ┌──────────────┐
    │"fire" = [0.5,│          │PE(1) = [0.84,│            │[1.34, 1.04,  │
    │  0.8, 0.2,   │    +     │ -0.46, 0.04, │     =      │  0.24, 0.62, │
    │  0.3, -0.1]  │          │  0.32, 0.99] │            │  0.89]       │
    └──────────────┘          └──────────────┘            └──────────────┘

    ┌──────────────┐          ┌──────────────┐            ┌──────────────┐
    │"wall" = [0.1,│          │PE(2) = [0.91,│            │[1.01, 0.69,  │
    │  0.3, -0.5,  │    +     │ -0.91, 0.08, │     =      │ -0.42, 0.96, │
    │  0.6, 0.4]   │          │  0.36, 0.54] │            │  0.94]       │
    └──────────────┘          └──────────────┘            └──────────────┘


    SINUSOIDAL PATTERN ACROSS POSITIONS:
    ════════════════════════════════════════════════════

    Position │  dim 0 (sin)  │  dim 1 (cos)  │  dim 2 (sin)  │  dim 3 (cos)
    ─────────┼───────────────┼───────────────┼───────────────┼──────────────
       0     │  0.000 ░      │  1.000 ████   │  0.000 ░      │  1.000 ████
       1     │  0.841 ███    │  0.540 ██     │  0.010 ░      │  0.999 ████
       2     │  0.909 ████   │ -0.416 ██     │  0.020 ░      │  0.999 ████
       3     │  0.141 ░      │ -0.990 ████   │  0.030 ░      │  0.999 ████
       4     │ -0.757 ███    │ -0.654 ███    │  0.040 ░      │  0.999 ████
       5     │ -0.959 ████   │  0.284 █      │  0.050 ░      │  0.999 ████
       6     │ -0.279 █      │  0.960 ████   │  0.060 ░      │  0.998 ████
       7     │  0.657 ███    │  0.754 ███    │  0.070 ░      │  0.998 ████

    Low dimensions → HIGH frequency (rapid changes across positions)
    High dimensions → LOW frequency (slow changes across positions)

    This creates a UNIQUE fingerprint for each position!


    WHY SINUSOIDAL?
    ════════════════════════════════════════════════════

    1. Each position gets a unique encoding vector
    2. The model can learn relative positions because:
       PE(pos+k) can be expressed as a linear function of PE(pos)
    3. Generalizes to sequence lengths not seen during training

    Position 0:  ∿∿∿∿∿∿∿∿  (phase 0)
    Position 1:  ∿∿∿∿∿∿∿∿  (shifted phase)
    Position 2:  ∿∿∿∿∿∿∿∿  (further shifted)
       ...
    Each position has a unique wave pattern fingerprint
```

### 4.3 Positional Encoding Heatmap

```
    POSITIONAL ENCODING HEATMAP (8 positions × 16 dimensions)
    ════════════════════════════════════════════════════════════

    Dimension →   0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
    Pos ↓      ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
      0        │░░░░│████│░░░░│████│░░░░│████│░░░░│████│░░░░│████│░░░░│████│░░░░│████│░░░░│████│
      1        │████│██░░│░░░░│████│░░░░│████│░░░░│████│░░░░│████│░░░░│████│░░░░│████│░░░░│████│
      2        │████│░░░░│░░░░│████│░░░░│████│░░░░│████│░░░░│████│░░░░│████│░░░░│████│░░░░│████│
      3        │░░██│░░░░│░░░░│████│░░░░│████│░░░░│████│░░░░│████│░░░░│████│░░░░│████│░░░░│████│
      4        │░░░░│░░░░│░░░█│████│░░░░│████│░░░░│████│░░░░│████│░░░░│████│░░░░│████│░░░░│████│
      5        │░░░░│░░██│████│███░│░░░░│████│░░░░│████│░░░░│████│░░░░│████│░░░░│████│░░░░│████│
      6        │░░░░│████│████│░░░░│░░░█│████│░░░░│████│░░░░│████│░░░░│████│░░░░│████│░░░░│████│
      7        │██░░│████│████│░░░░│████│███░│░░░░│████│░░░░│████│░░░░│████│░░░░│████│░░░░│████│
               └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘

    ████ = positive values (close to +1)    ░░░░ = values near zero or negative

    Notice: Left columns (low dims) change rapidly → high frequency
            Right columns (high dims) change slowly → low frequency
```

---

## 5. Encoder and Decoder Blocks in Detail

### 5.1 Single Encoder Block

```
    ENCODER BLOCK (repeated N times, typically N=6)
    ════════════════════════════════════════════════════

                    ┌───────────────────┐
                    │      Output       │
                    └─────────┬─────────┘
                              │
                    ┌─────────┴─────────┐
                    │  Layer Norm       │
                ┌───┤                   │
                │   └─────────┬─────────┘
                │             │
                │   ┌─────────┴─────────┐
    Residual    │   │  Feed-Forward NN  │
    Connection──┤   │  FFN(x) = ReLU(  │
                │   │   xW₁+b₁)W₂+b₂  │
                │   └─────────┬─────────┘
                │             │
                └──────►(+)◄──┘
                         │
                    ┌────┴──────────────┐
                    │  Layer Norm       │
                ┌───┤                   │
                │   └─────────┬─────────┘
                │             │
                │   ┌─────────┴─────────┐
    Residual    │   │  Multi-Head       │
    Connection──┤   │  Self-Attention   │
                │   │  (Q=K=V=input)    │
                │   └─────────┬─────────┘
                │             │
                └──────►(+)◄──┘
                         │
                    ┌────┴──────────────┐
                    │  Input + Pos Enc  │
                    └───────────────────┘
```

### 5.2 Single Decoder Block

```
    DECODER BLOCK (repeated N times, typically N=6)
    ════════════════════════════════════════════════════

                    ┌───────────────────┐
                    │      Output       │
                    └─────────┬─────────┘
                              │
                    ┌─────────┴─────────┐
                    │  Layer Norm + Add │
                    └─────────┬─────────┘
                              │
                    ┌─────────┴─────────┐
                    │  Feed-Forward NN  │
                    └─────────┬─────────┘
                              │
                    ┌─────────┴─────────┐
                    │  Layer Norm + Add │
                    └─────────┬─────────┘
                              │
                    ┌─────────┴─────────┐
                    │  Multi-Head       │
                    │  Cross-Attention  │◄──── K, V from Encoder
                    │  (Q from decoder) │
                    └─────────┬─────────┘
                              │
                    ┌─────────┴─────────┐
                    │  Layer Norm + Add │
                    └─────────┬─────────┘
                              │
                    ┌─────────┴─────────┐
                    │  Masked Multi-Head│
                    │  Self-Attention   │
                    │  (prevents future │
                    │   token peeking)  │
                    └─────────┬─────────┘
                              │
                    ┌─────────┴─────────┐
                    │Target + Pos Enc   │
                    └───────────────────┘
```

---

## 6. Transformer vs. RNN/LSTM Comparison

```
    ════════════════════════════════════════════════════════════════
    │  Feature           │  RNN/LSTM          │  Transformer      │
    ════════════════════════════════════════════════════════════════
    │  Processing        │  Sequential ───→   │  Parallel ═══►    │
    │  Long Dependencies │  Weak (vanishing   │  Strong (direct   │
    │                    │  gradient)         │  attention)       │
    │  Training Speed    │  Slow (sequential) │  Fast (parallel)  │
    │  Memory            │  Fixed hidden state│  Attends to all   │
    │  Scalability       │  Limited           │  Highly scalable  │
    ════════════════════════════════════════════════════════════════

    RNN Processing:
    x₁ → [h₁] → x₂ → [h₂] → x₃ → [h₃] → x₄ → [h₄]
    (must wait for each step)

    Transformer Processing:
    x₁ ═══╗
    x₂ ═══╬═══► [Self-Attention] ═══► Output (all at once!)
    x₃ ═══╣
    x₄ ═══╝
```

---

## 7. Applications of Transformers in Cybersecurity

Transformers have become increasingly vital in cybersecurity due to their ability to process sequential data, understand context, and detect anomalous patterns.

### 7.1 Key Cybersecurity Applications

```
    TRANSFORMER APPLICATIONS IN CYBERSECURITY
    ════════════════════════════════════════════════════════════════

    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │  1. MALWARE     │     │  2. INTRUSION   │     │  3. PHISHING    │
    │  DETECTION      │     │  DETECTION      │     │  DETECTION      │
    │                 │     │                 │     │                 │
    │  Analyze API    │     │  Process network│     │  Analyze email  │
    │  call sequences │     │  packet flows   │     │  text and URLs  │
    │  and binary     │     │  to identify    │     │  to classify    │
    │  patterns       │     │  attack traffic │     │  threats        │
    └─────────────────┘     └─────────────────┘     └─────────────────┘

    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │  4. LOG         │     │  5. VULNERABILITY│    │  6. THREAT      │
    │  ANALYSIS       │     │  DETECTION       │    │  INTELLIGENCE   │
    │                 │     │                 │     │                 │
    │  Parse system   │     │  Analyze source │     │  Process CTI    │
    │  logs to detect │     │  code for       │     │  reports and    │
    │  anomalies and  │     │  security flaws │     │  dark web data  │
    │  incidents      │     │  automatically  │     │  for insights   │
    └─────────────────┘     └─────────────────┘     └─────────────────┘
```

### 7.2 Detailed Application Examples

**1. Malware Detection:** Transformers can process sequences of API calls, system calls, or byte-level representations of executables. The self-attention mechanism captures dependencies between distant operations that may indicate malicious behavior (e.g., a file operation followed much later by a network call to exfiltrate data).

**2. Network Intrusion Detection (NIDS):** Network traffic can be represented as sequences of packet features. Transformers analyze temporal patterns across packet flows, identifying sophisticated multi-stage attacks (APTs) that traditional signature-based systems miss.

**3. Phishing Detection:** Transformer-based models like BERT can analyze email content, subject lines, and embedded URLs to distinguish phishing attempts from legitimate communications with high accuracy, understanding the subtle linguistic cues that indicate social engineering.

**4. Log Analysis & SIEM:** Security Information and Event Management systems generate massive volumes of logs. Transformers can process these sequential log entries to detect anomalous patterns, correlate events across different systems, and identify security incidents in real-time.

**5. Vulnerability Detection:** Models like CodeBERT apply Transformer architectures to source code analysis, identifying common vulnerability patterns (SQL injection, buffer overflow, XSS) by understanding code semantics and context.

**6. Threat Intelligence:** Transformers process and correlate unstructured threat intelligence reports, CVE descriptions, and dark web forum posts to extract indicators of compromise (IoCs) and predict emerging threats.

### 7.3 Why Transformers Excel in Cybersecurity

```
    WHY TRANSFORMERS FOR CYBERSECURITY?
    ════════════════════════════════════════════════════

    Traditional ML                  Transformer-Based
    ┌──────────────────┐           ┌──────────────────┐
    │ Fixed feature     │           │ Learns features  │
    │ engineering       │           │ automatically    │
    │                   │           │                  │
    │ Misses long-range │           │ Captures distant │
    │ attack patterns   │           │ dependencies     │
    │                   │           │                  │
    │ Cannot understand │           │ Understands      │
    │ context/semantics │           │ context deeply   │
    │                   │           │                  │
    │ Struggles with    │           │ Processes        │
    │ variable-length   │           │ any length       │
    │ sequences         │           │ sequences        │
    │                   │           │                  │
    │ Single-task       │           │ Transfer learning│
    │ models            │           │ (pre-train once, │
    │                   │           │  fine-tune many) │
    └──────────────────┘           └──────────────────┘
```

---

## 8. Practical Cybersecurity Example: Phishing URL Detection

Below is a simplified demonstration of how a Transformer-based approach can be used to classify URLs as phishing or legitimate by treating URL characters as a sequence:

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Embedding,
    GlobalAveragePooling1D, LayerNormalization, MultiHeadAttention
)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ============================================================
# 1. SAMPLE PHISHING & LEGITIMATE URL DATA
# ============================================================
# In production, you would use datasets like PhishTank or OpenPhish
urls = [
    # Legitimate URLs (label = 0)
    "https://www.google.com/search?q=python",
    "https://github.com/tensorflow/tensorflow",
    "https://stackoverflow.com/questions/tagged/python",
    "https://www.wikipedia.org/wiki/Machine_learning",
    "https://docs.python.org/3/library/os.html",
    "https://www.amazon.com/dp/B08N5WRWNW",
    "https://mail.google.com/mail/inbox",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://www.linkedin.com/in/johndoe",
    "https://www.reddit.com/r/cybersecurity",
    "https://news.ycombinator.com",
    "https://www.microsoft.com/en-us/windows",
    "https://www.apple.com/iphone",
    "https://www.netflix.com/browse",
    "https://www.nytimes.com/section/technology",
    "https://www.bbc.com/news/technology",
    "https://www.dropbox.com/home",
    "https://slack.com/intl/en-us",
    "https://www.spotify.com/account/overview",
    "https://www.paypal.com/myaccount/summary",
    # Phishing URLs (label = 1)
    "http://gooogle-login.secure-verify.xyz/signin",
    "http://192.168.1.1/paypal-confirm/login.html",
    "http://amaz0n-security.tk/verify-account",
    "http://secure-bankofamerica.login.phishsite.ru/auth",
    "http://microsoft-365-verify.suspicious-domain.cn/reset",
    "http://appleid.apple.com.verify.evil.com/login",
    "http://dropbox-sharing.malware-site.tk/download",
    "http://linkedin-profile.verify-now.xyz/update",
    "http://netflix-payment.update-billing.ru/form",
    "http://facebook-security.account-verify.tk/confirm",
    "http://paypa1-secure.login-verify.xyz/account",
    "http://g00gle-drive.share-document.cn/view",
    "http://twitter-verify.suspicious.tk/badge",
    "http://instagram-help.account-recovery.xyz/reset",
    "http://wellsfarg0-secure.banking-login.ru/online",
    "http://chase-verify.secure-banking.tk/logon",
    "http://icloud-find.apple-verify.xyz/locate",
    "http://outlook-365.microsoft-reset.cn/password",
    "http://github-security.verify-account.tk/2fa",
    "http://slack-workspace.join-verify.xyz/invite",
]

labels = [0]*20 + [1]*20  # 0 = legitimate, 1 = phishing

# ============================================================
# 2. URL CHARACTER-LEVEL TOKENIZATION
# ============================================================
# Create character vocabulary
all_chars = set("".join(urls))
char_to_idx = {c: i+1 for i, c in enumerate(sorted(all_chars))}
vocab_size = len(char_to_idx) + 1
max_len = 80  # Maximum URL length

# Encode URLs as character sequences
def encode_url(url, char_map, maxlen):
    encoded = [char_map.get(c, 0) for c in url[:maxlen]]
    return encoded

X = [encode_url(url, char_to_idx, max_len) for url in urls]
X = pad_sequences(X, maxlen=max_len, padding='post')
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Vocabulary size: {vocab_size}")
print(f"Max URL length:  {max_len}")
print(f"Training set:    {len(X_train)} samples")
print(f"Test set:        {len(X_test)} samples")

# ============================================================
# 3. POSITIONAL ENCODING LAYER
# ============================================================
class PositionalEncoding(tf.keras.layers.Layer):
    """Sinusoidal positional encoding as described in
    'Attention Is All You Need'"""

    def __init__(self, max_len, d_model):
        super().__init__()
        # Compute positional encodings
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)

    def call(self, x):
        return x + self.pe[:, :tf.shape(x)[1], :]

# ============================================================
# 4. BUILD TRANSFORMER MODEL
# ============================================================
d_model = 64
num_heads = 4
ff_dim = 128

# Input
inputs = Input(shape=(max_len,))

# Embedding + Positional Encoding
x = Embedding(input_dim=vocab_size, output_dim=d_model)(inputs)
x = PositionalEncoding(max_len, d_model)(x)

# Transformer Encoder Block
attn_output = MultiHeadAttention(
    num_heads=num_heads, key_dim=d_model // num_heads
)(x, x)
attn_output = Dropout(0.1)(attn_output)
x = LayerNormalization(epsilon=1e-6)(x + attn_output)

ff_output = Dense(ff_dim, activation='relu')(x)
ff_output = Dense(d_model)(ff_output)
ff_output = Dropout(0.1)(ff_output)
x = LayerNormalization(epsilon=1e-6)(x + ff_output)

# Classification Head
x = GlobalAveragePooling1D()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================================
# 5. TRAIN THE MODEL
# ============================================================
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=8,
    validation_split=0.2,
    verbose=1
)

# ============================================================
# 6. EVALUATE
# ============================================================
y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=['Legitimate', 'Phishing']
))

# ============================================================
# 7. TEST ON NEW URLs
# ============================================================
test_urls = [
    "https://www.google.com",
    "http://g00gle-verify.malicious-site.tk/login",
    "https://www.github.com/user/repo",
    "http://paypa1-security.verify-now.xyz/account"
]

for url in test_urls:
    encoded = pad_sequences(
        [encode_url(url, char_to_idx, max_len)],
        maxlen=max_len, padding='post'
    )
    pred = model.predict(encoded, verbose=0)[0][0]
    label = "PHISHING" if pred > 0.5 else "LEGITIMATE"
    print(f"  {label} ({pred:.3f}): {url}")
```

---

## 9. Summary

The Transformer architecture has fundamentally changed the landscape of deep learning and is increasingly being adopted in cybersecurity. Its core innovations — **self-attention** for capturing dependencies between any elements in a sequence, **positional encoding** for maintaining order information, and **multi-head attention** for learning diverse relationship patterns — make it exceptionally well-suited for security tasks that involve sequential and contextual data.

From detecting phishing URLs by analyzing character-level patterns, to identifying network intrusions by processing traffic flows, to analyzing malware behavior through API call sequences, Transformers provide a powerful, flexible, and scalable framework for building next-generation cybersecurity defenses.

As cyber threats grow more sophisticated, the ability of Transformers to understand context, learn from vast amounts of data through transfer learning, and adapt to new threat patterns positions them as an indispensable tool in the modern cybersecurity arsenal.

---

*References: Vaswani et al. (2017) "Attention Is All You Need"; Devlin et al. (2019) "BERT: Pre-training of Deep Bidirectional Transformers"; Brown et al. (2020) "Language Models are Few-Shot Learners (GPT-3)"; Ferrag et al. (2023) "Transformers in Cybersecurity: A Survey"*
