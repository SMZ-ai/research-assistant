# Mixture of Experts — Architecture Analysis & Comparison
## Multi-Agent Research Assistant — Course 606363
### University of Petra | Data Science and Artificial Intelligence

---

## 1. Introduction to Mixture of Experts

The Mixture of Experts (MoE) architecture is a machine learning paradigm where a model consists of multiple specialized sub-networks — called **experts** — alongside a **gating network** (or router) that decides which expert(s) to activate for each input. Rather than using all model parameters for every computation, MoE selectively routes each token to only a subset of experts, dramatically improving computational efficiency.

The core insight is: **not every input requires the same computation**. A question about mathematics should activate different neural circuits than a question about poetry. MoE formalizes this intuition into an architecture that scales model capacity without proportionally scaling compute cost.

The fundamental MoE formula is:

```
MoE(x) = Σ G(x)_i × E_i(x)
```

Where:
- `x` is the input token
- `G(x)` is the gating function output (a probability distribution over experts)
- `E_i(x)` is the output of expert i
- Only the top-K experts with highest gate values are activated

---

## 2. Key MoE Concepts

### 2.1 Sparse vs. Dense Activation

**Dense models** (standard transformers like Llama, GPT-4) activate all parameters for every token. A 70B parameter model uses all 70B parameters per forward pass.

**Sparse MoE models** activate only a fraction. Mixtral 8x7B has 47B total parameters but activates only ~13B per token — using 8 experts per FFN layer but only routing each token to 2 of them.

This creates an important distinction:
- **Total parameters** = model size on disk / in memory
- **Active parameters** = compute per token (determines speed and cost)

### 2.2 Top-K Routing

The most common routing strategy selects the top-K experts for each token based on gating scores. For K=2 (used in Mixtral):

```
G(x) = Softmax(TopK(x · W_g, K))
```

Only the top-2 experts receive non-zero weights. All others contribute nothing to the output. This sparsity is what makes MoE computationally efficient.

### 2.3 Load Balancing

A critical challenge in MoE is **expert collapse** — where the router learns to always send tokens to the same 1-2 experts, leaving others untrained. Solutions include:

- **Auxiliary loss** — penalizes uneven expert utilization during training
- **Expert capacity** — limits how many tokens each expert can process per batch
- **Random routing noise** — adds noise during training to encourage exploration

---

## 3. Architecture Comparison: Three Major MoE Models

### 3.1 Switch Transformer (Google, 2021)

**Paper**: "Switch Transformers: Scaling to Trillion Parameter Models" — Fedus et al.

**Key design choices:**
- Routes each token to exactly **K=1 expert** (simplest possible MoE)
- Simplifies load balancing significantly
- Uses a capacity factor to handle token overflow
- Demonstrated scaling to 1.6 trillion parameters

**Architecture details:**
- Replaces every FFN layer in the transformer with an MoE layer
- 2,048 experts in the largest configuration
- Expert capacity = (tokens_per_batch / num_experts) × capacity_factor

**Advantages:**
- Simplest routing — easiest to train stably
- Extreme parameter scaling at low compute cost
- Strong performance on language tasks

**Disadvantages:**
- Top-1 routing loses information (only one expert's perspective)
- More sensitive to expert collapse than Top-2
- Less sample efficient than dense models at small scale

**Performance benchmark (T5 comparison):**
| Model | Parameters | FLOP/token | Quality |
|---|---|---|---|
| T5-Base (dense) | 0.2B | High | Baseline |
| Switch-Base | 7B | Same as T5-Base | +5% |
| Switch-Large | 26B | Same as T5-Large | +4% |

---

### 3.2 Mixtral 8x7B (Mistral AI, 2024)

**Paper**: "Mixtral of Experts" — Mistral AI

**Key design choices:**
- Routes each token to exactly **K=2 experts** out of 8
- Sliding Window Attention for long contexts
- 32,768 token context window
- Fully open-source weights

**Architecture details:**
- 8 experts per FFN layer (replaces standard FFN)
- 32 transformer layers, each with an MoE FFN
- Hidden dimension: 4,096
- Total parameters: ~47B
- Active parameters per token: ~13B (roughly equivalent to a 13B dense model)

**The 8x7B naming** refers to 8 experts each with 7B parameters in the FFN layer — though total model size is ~47B due to shared attention layers.

**Routing mechanism:**
```python
# Simplified Mixtral routing
router_logits = token_embedding @ router_weights  # [batch, seq, num_experts]
top2_experts = torch.topk(router_logits, k=2, dim=-1)
weights = F.softmax(top2_experts.values, dim=-1)
output = sum(weights[i] * expert_i(token) for i in top2_indices)
```

**Advantages:**
- Excellent quality-to-compute ratio
- Open weights — fully reproducible
- Strong on reasoning and coding benchmarks
- Efficient inference with only 13B active params

**Disadvantages:**
- Still requires ~90GB VRAM to load all parameters
- More complex training than dense models
- Load balancing remains challenging

**Performance benchmarks (vs dense models):**
| Benchmark | Llama 2 70B | Mixtral 8x7B |
|---|---|---|
| MMLU | 69.8% | 70.6% |
| HumanEval | 29.9% | 40.2% |
| Math | 13.5% | 28.4% |
| MBPP | 49.8% | 60.7% |

Mixtral matches or exceeds Llama 2 70B on most benchmarks while using ~6x less compute per token.

---

### 3.3 DeepSeek-MoE (DeepSeek AI, 2024)

**Paper**: "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models"

**Key design choices:**
- **Fine-grained expert segmentation** — splits experts into smaller units for better specialization
- **Shared experts** — keeps some experts always active for general knowledge
- **Top-K routing with K much larger than 2** — selects from more fine-grained experts
- Achieves same performance as Mixtral with 40% less compute

**Architecture innovation:**

DeepSeek-MoE introduces two types of experts:
1. **Shared experts (Ks)** — always activated, handle general/common knowledge
2. **Routed experts (Kr)** — selectively activated based on input

```
Output = Shared_Expert(x) + Σ(top-K routing from Routed_Experts)
```

This separation prevents the router from wasting capacity routing common knowledge through expensive selection, reserving routing decisions for specialized information.

**Fine-grained segmentation:**
Instead of 8 large experts, DeepSeek uses 64 smaller experts — each expert is 1/8 the size. The router then selects top-16 from 64. This enables much more fine-grained specialization while maintaining similar compute.

**Performance comparison:**
| Model | Active Params | Total Params | MMLU |
|---|---|---|---|
| LLaMA 2 7B (dense) | 7B | 7B | 45.3% |
| Mixtral 8x7B | 13B | 47B | 70.6% |
| DeepSeek-MoE 16B | 2.8B | 16B | 45.0% |
| DeepSeek-V2 236B | 21B | 236B | 78.5% |

DeepSeek-MoE 16B achieves comparable quality to LLaMA 2 7B using only 2.8B active parameters — a 60% compute reduction.

---

## 4. Head-to-Head Comparison

| Feature | Switch Transformer | Mixtral 8x7B | DeepSeek-MoE |
|---|---|---|---|
| Year | 2021 | 2024 | 2024 |
| Developer | Google | Mistral AI | DeepSeek AI |
| Routing | Top-1 | Top-2 of 8 | Top-K of fine-grained |
| Shared experts | No | No | Yes |
| Open source | Yes (weights limited) | Yes (fully open) | Yes |
| Context window | 512 tokens | 32,768 tokens | 4,096 tokens |
| Best for | Maximum scale | Balanced performance | Compute efficiency |
| Training stability | High | Medium | High |
| Expert collapse risk | High | Medium | Low |

---

## 5. MoE vs. Dense Models: Trade-offs

### When MoE Wins
- **Large-scale training**: MoE enables 10x more parameters at the same compute budget
- **Diverse tasks**: Different experts specialize in different domains
- **Inference at scale**: Once loaded, MoE is faster per token than a dense model of equal quality

### When Dense Models Win
- **Small scale**: MoE training instability hurts at small model sizes
- **Memory-constrained deployment**: All expert weights must be in memory
- **Fine-tuning**: Dense models are easier to fine-tune on domain-specific data
- **Latency-sensitive applications**: Expert routing adds overhead

### The Fundamental Trade-off

```
Dense model quality at 70B params requires 70B active params per token
MoE model quality at 70B params requires only ~13B active params per token
→ MoE is ~5x more efficient at inference, but needs more memory
```

---

## 6. Integration in Our Project

Our multi-agent system demonstrates MoE awareness in two ways:

**1. Using Groq's optimized inference for Llama 3.3-70b** — Groq's LPU (Language Processing Unit) hardware is designed specifically to accelerate transformer inference, including sparse attention patterns similar to MoE routing.

**2. MoE analysis embedded in the system state:**
```python
class ResearchState(TypedDict):
    moe_analysis: Optional[dict]  # MoE model comparison results
    transformer_config: dict      # LLM model details
```

The system is architected to support swapping Llama for Mixtral or DeepSeek-V2 — any LangChain-compatible model works with our agent framework. A future enhancement would implement a dynamic router that selects between multiple LLMs based on the task type (e.g., use a code-specialized model for technical research).

---

## 7. Future Directions in MoE

- **Expert merging** — combining trained experts to reduce memory footprint
- **Conditional compute** — varying the number of active experts per token based on difficulty
- **Hierarchical MoE** — routing at multiple levels of granularity
- **Cross-layer MoE** — sharing experts across transformer layers
- **MoE for multimodal models** — routing between vision and language experts

---

## 8. References

1. Fedus, W., Zoph, B., & Shazeer, N. (2021). "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity." *JMLR 2022*.
2. Mistral AI. (2024). "Mixtral of Experts." *arXiv:2401.04088*.
3. Dai, D., et al. (2024). "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models." *arXiv:2401.06066*.
4. Shazeer, N., et al. (2017). "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer." *ICLR 2017*.
5. Lepikhin, D., et al. (2021). "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding." *ICLR 2021*.
6. Zoph, B., et al. (2022). "ST-MoE: Designing Stable and Transferable Sparse Expert Models." *arXiv:2202.08906*.
