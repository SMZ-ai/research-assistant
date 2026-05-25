# Transformer Architecture Analysis
## Multi-Agent Research Assistant — Course 606363
### University of Petra | Data Science and Artificial Intelligence

---

## 1. Introduction

The transformer architecture, introduced by Vaswani et al. in the landmark 2017 paper "Attention Is All You Need," represents one of the most significant breakthroughs in the history of artificial intelligence. Prior to transformers, sequence modeling relied heavily on recurrent neural networks (RNNs) and long short-term memory networks (LSTMs), which processed tokens sequentially and struggled with long-range dependencies. The transformer eliminated recurrence entirely, replacing it with a self-attention mechanism that processes all tokens in parallel — a design choice that enabled unprecedented scaling and performance gains.

This project uses **Llama 3.3-70b**, a transformer-based large language model developed by Meta AI, accessed via the Groq API. The following analysis documents the transformer architecture underlying this model and explains how it powers each agent in our multi-agent research pipeline.

---

## 2. Core Transformer Architecture

### 2.1 High-Level Structure

The transformer follows an **encoder-decoder** architecture in its original form. However, modern large language models like Llama use a **decoder-only** architecture, which is optimized for text generation tasks. The decoder-only design takes an input sequence and autoregressively generates the next token, one at a time, based on all previously seen tokens.

The key components of the transformer are:

- **Token Embedding Layer** — Converts raw text tokens into dense vector representations
- **Positional Encoding** — Injects position information since transformers have no inherent sense of order
- **Multi-Head Self-Attention** — The core mechanism that allows each token to attend to all other tokens
- **Feed-Forward Network (FFN)** — Applies non-linear transformations to each token independently
- **Layer Normalization** — Stabilizes training by normalizing activations
- **Output Projection** — Maps the final hidden states to vocabulary probabilities

### 2.2 Self-Attention Mechanism

Self-attention is the heart of the transformer. For each token in the sequence, the mechanism computes three vectors:

- **Query (Q)** — What this token is looking for
- **Key (K)** — What this token offers to others
- **Value (V)** — The actual content this token contributes

The attention score between two tokens is computed as:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

Where `d_k` is the dimension of the key vectors. The scaling by `√d_k` prevents the dot products from becoming too large, which would push the softmax into regions with very small gradients.

The softmax operation converts the raw scores into a probability distribution, which becomes the **attention weight** — how much each token attends to every other token. This allows the model to capture long-range dependencies that RNNs struggle with.

### 2.3 Multi-Head Attention

Instead of computing a single attention function, transformers run attention multiple times in parallel with different learned projections — these are called **attention heads**. Each head can focus on different aspects of the relationship between tokens:

- One head might focus on syntactic relationships (subject-verb agreement)
- Another might capture semantic similarity
- Another might track co-reference (pronouns referring back to nouns)

The outputs of all heads are concatenated and linearly projected back to the model dimension:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_O
```

Llama 3.3-70b uses **Grouped Query Attention (GQA)**, a variant where multiple query heads share the same key and value heads. This significantly reduces memory usage during inference while maintaining model quality.

### 2.4 Feed-Forward Network

After each attention layer, every token passes through a position-wise feed-forward network independently. This is a two-layer MLP with a non-linear activation function:

```
FFN(x) = max(0, xW_1 + b_1) W_2 + b_2
```

Llama uses the **SwiGLU** activation function instead of ReLU, which has been shown to improve model performance:

```
SwiGLU(x, W, V, b, c) = Swish(xW + b) ⊙ (xV + c)
```

The FFN layers make up approximately two-thirds of the model parameters and are responsible for storing factual knowledge and performing complex reasoning.

### 2.5 Positional Encoding

Since self-attention treats all positions equally, position information must be injected. Llama 3.3-70b uses **Rotary Positional Embedding (RoPE)**, which encodes position information directly into the query and key vectors using rotation matrices. RoPE has the advantage of naturally extending to sequence lengths longer than those seen during training.

### 2.6 Layer Normalization

Llama uses **RMSNorm** (Root Mean Square Layer Normalization) instead of the standard LayerNorm. RMSNorm normalizes only by the root mean square of the activations, without centering, which is computationally cheaper and empirically performs equally well.

---

## 3. Llama 3.3-70b Specifications

| Parameter | Value |
|---|---|
| Total parameters | 70 billion |
| Architecture | Decoder-only transformer |
| Attention type | Grouped Query Attention (GQA) |
| Context length | 128,000 tokens |
| Vocabulary size | 128,256 tokens |
| Number of layers | 80 |
| Hidden dimension | 8,192 |
| Attention heads | 64 (query), 8 (key/value) |
| FFN activation | SwiGLU |
| Positional encoding | RoPE |
| Normalization | RMSNorm |
| Training data | ~15 trillion tokens |
| Released by | Meta AI (2024) |

---

## 4. How the Transformer Powers Our System

Each agent in our multi-agent pipeline uses the transformer differently:

### Research Agent
Uses the transformer's **language understanding** capabilities to generate targeted search queries from a research topic. The model reasons about what information is needed and formulates precise queries.

### Classification Agent
Uses the transformer's **text classification** capabilities to categorize sources by type (academic paper, blog, news) and relevance. The model reads the title and content snippet and outputs a structured JSON classification.

### NER Agent
Uses the transformer's **named entity recognition** capabilities to identify people, organizations, technologies, and dates that SpaCy's rule-based system might miss — especially domain-specific AI terms.

### Analyzer Agent
Uses the transformer's **reasoning capabilities** (Chain-of-Thought) to identify themes and patterns across multiple sources. The model synthesizes information from many documents into a coherent analytical structure.

### Writer Agent
Uses the transformer's **text generation** capabilities to write full academic prose. This is the most demanding use — the model must maintain coherence across thousands of tokens while following a structured outline and integrating citations.

### Critic Agent
Uses the transformer's **evaluation capabilities** to assess report quality using the Reflection pattern. The model scores the draft on multiple dimensions and provides actionable feedback.

---

## 5. Key Architectural Innovations in Modern LLMs

### 5.1 Scaling Laws
Research by Kaplan et al. (2020) showed that transformer performance scales predictably with model size, data size, and compute. This led to the strategy of training ever-larger models, culminating in systems like GPT-4, Claude, and Llama.

### 5.2 Instruction Fine-tuning
Raw transformer models trained only on next-token prediction are not immediately useful as assistants. Models like Llama 3.3 undergo **instruction fine-tuning** using RLHF (Reinforcement Learning from Human Feedback) to align with human preferences and follow instructions.

### 5.3 Flash Attention
Standard attention has quadratic complexity O(n²) in sequence length. Flash Attention (Dao et al., 2022) reorganizes the computation to be IO-aware, significantly reducing memory usage and enabling longer context windows.

### 5.4 KV Cache
During autoregressive generation, the key and value vectors for all previous tokens must be recomputed at each step. The **KV cache** stores these vectors so they only need to be computed once, dramatically speeding up inference.

---

## 6. Transformer vs. Previous Architectures

| Feature | RNN/LSTM | Transformer |
|---|---|---|
| Processing order | Sequential | Parallel |
| Long-range dependencies | Difficult | Natural (via attention) |
| Training speed | Slow | Fast (parallelizable) |
| Context length | Limited | Very long (128K+ tokens) |
| Scalability | Limited | Scales with compute |
| State | Hidden state | No persistent state |

---

## 7. Model Configuration in Our Project

In our system, the transformer is configured differently per agent to balance quality and speed:

```python
# Research, Classification, NER, Analyzer, Critic — fast responses
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,        # deterministic output
    groq_api_key=os.getenv('GROQ_API_KEY'),
)

# Writer Agent — creative, longer output
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.4,      # slightly creative
    max_tokens=4096,      # long output needed
    groq_api_key=os.getenv('GROQ_API_KEY'),
)
```

**Temperature** controls randomness: 0 = deterministic (best for JSON output), 0.4 = slightly creative (best for writing prose).

---

## 8. References

1. Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS 2017*.
2. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *NAACL 2019*.
3. Meta AI. (2024). "Llama 3 Technical Report." Meta AI Research.
4. Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *NeurIPS 2022*.
5. Kaplan, J., et al. (2020). "Scaling Laws for Neural Language Models." *arXiv:2001.08361*.
6. Su, J., et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding." *arXiv:2104.09864*.
7. Ainslie, J., et al. (2023). "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." *EMNLP 2023*.
