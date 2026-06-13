# Parallelism Explorer — Modeling Methodology

This document explains how each aspect of the distributed training simulator was modeled, the assumptions made, and where the numbers come from.

---

## 1. Memory Model

### 1.1 Training Memory per Parameter

Training with mixed-precision (AMP) and Adam optimizer requires storing multiple copies of each parameter:

| Component | Bytes/param (FP16 AMP) | Bytes/param (FP32) | Notes |
|-----------|----------------------|-------------------|-------|
| Parameters | 2 | 4 | Working copy for forward/backward |
| Gradients | 2 | 4 | Accumulated during backward pass |
| Adam master weights | 4 | — | FP32 copy for numeric stability |
| Adam momentum (m) | 4 | 4 | First moment estimate |
| Adam variance (v) | 4 | 4 | Second moment estimate |
| **Total** | **16** | **16** | Same total, different breakdown |

In the simulator: `paramGB = P × 2`, `gradGB = P × 2`, `optimGB = P × 12` (for mixed precision). These are in billions of params × bytes, giving GB directly.

### 1.2 Activation Memory

Activations are the intermediate tensors stored during forward pass for use in backpropagation. The formula used:

```
activation_GB ≈ batch_size × seq_len × hidden_dim × num_layers × 8 / 1e9
```

The factor of 8 (bytes) is a simplified proxy for the multiple tensors retained per layer: attention QKV projections, attention scores, MLP intermediates, layer norm inputs, and residual connections. In reality, the exact factor is architecture-dependent (roughly 34 × 2 bytes per element for a standard transformer), but 8 bytes/element × (hidden_dim elements per position) gives a reasonable order-of-magnitude estimate when multiplied across layers.

**With activation checkpointing**: Only `√L` evenly-spaced layer activations are stored. During backward, the intermediate activations are recomputed from the nearest checkpoint. This changes the formula to use `√layers` instead of `layers`, reducing memory at the cost of ~33% additional compute.

### 1.3 Inference Memory

Inference has no gradients or optimizer states. The dominant memory consumers are:

- **Parameters**: `P × 2 bytes` (fp16)
- **KV Cache**: `2 × layers × hidden_dim × seq_len × batch_size × 2 bytes`

The KV cache stores key and value projections for all previous tokens across all attention layers. The factor of 2 accounts for both K and V tensors. This grows linearly with sequence length and batch size, and is often the binding constraint for inference serving.

### 1.4 How Each Paradigm Distributes Memory

The simulator uses a divisor model `d(param_div, grad_div, optim_div, act_div)` to express how each paradigm shards the four memory components:

| Paradigm | Params ÷ | Grads ÷ | Optimizer ÷ | Activations ÷ |
|----------|----------|---------|-------------|----------------|
| DDP | 1 | 1 | 1 | N |
| FSDP | N | N | N | N |
| PP | N | N | N | N |
| TP | N | N | N | 1 |
| 3D | TP×PP | TP×PP | TP×PP×DP | DP |

Key observations:
- **DDP replicates everything** except data (each GPU processes a different mini-batch, so activations are divided by N).
- **FSDP (ZeRO-3) shards everything** evenly — the hallmark advantage for large models.
- **PP splits by layers**, so each GPU holds 1/N of the model, gradients, and optimizer for its assigned layers.
- **TP splits weight matrices** but activations are replicated (all GPUs see the same input tensor and need to store their portion of intermediate results, but the full activation tensor is used in each layer).
- **3D** compounds: TP×PP handles model sharding, DP further shards optimizer states (via FSDP on the DP dimension).

---

## 2. Hardware FLOP Utilization (HFU)

### 2.1 Definition

HFU measures what fraction of the GPU's theoretical peak FLOPS is achieved during training:

```
HFU = Achieved TFLOPS / Peak TFLOPS
```

Where achieved TFLOPS is derived from:
```
Achieved = Total_FLOPs / (wall_clock_time × num_GPUs)
```

### 2.2 Model FLOPs

For a transformer, the FLOPs per training iteration are approximately:

```
FLOPs = C × P × global_batch × seq_len
```

Where C = 6 for standard forward+backward (2 for forward matmuls, 4 for backward — backward is ~2× forward due to computing both weight and activation gradients). With activation checkpointing, C = 8 (extra forward pass for recomputation).

### 2.3 Estimation Model

Since the simulator doesn't run real training, HFU is estimated as a product of five independent efficiency factors:

```
HFU = compute_eff × comm_eff × pipe_eff × batch_eff × precision_factor
```

**Compute efficiency** (`compute_eff`): How well the model's matrix dimensions map to GPU tensor core tile sizes. Larger hidden dimensions (4096+) fill the 16×16 or 32×32 tensor core tiles more completely. We model this as:

```
compute_eff = min(0.68, 0.30 + H/24000 × 0.38)
```

This gives ~31% for GPT-2 (H=768), ~37% for LLaMA 7B (H=4096), and ~56% for LLaMA 405B (H=16384). Real-world values from published papers (PaLM, LLaMA) fall in this range.

**Communication efficiency** (`comm_eff`): Models the fraction of time spent in useful compute vs. waiting for collective operations. Paradigm-specific:

- DDP: `max(0.65, 1.0 - 0.035 × log₂(N))` — AllReduce overlaps well with backward computation via bucketed gradient communication. Degrades logarithmically with GPU count.
- FSDP: `max(0.50, 0.93 - 0.055 × log₂(N))` — More communication (AllGather + ReduceScatter per layer) with partial overlap. Higher base cost than DDP.
- PP: `max(0.80, 0.98 - 0.015 × log₂(N))` — Minimal communication (only activation tensors between adjacent stages). Pipeline bubbles are captured separately.
- TP: Depends on whether GPUs are intra-node (NVLink) or cross-node. Intra-node: `max(0.72, 0.95 - 0.03 × log₂(N))`. Cross-node: `max(0.45, 0.85 - 0.06 × log₂(N))`. The per-layer AllReduce makes TP highly bandwidth-sensitive.
- 3D: Product of TP communication (intra-node) and DP communication factors.

**Pipeline efficiency** (`pipe_eff`): The classic bubble fraction formula for pipeline parallelism:

```
pipe_eff = micro_batches / (micro_batches + stages - 1)
```

Where `micro_batches = max(stages, batch_size)`. This captures the fundamental tradeoff: more micro-batches fill the pipeline better, but the startup/drain cost is `(stages - 1)` idle slots. Only applies to PP and the PP dimension of 3D.

**Batch utilization** (`batch_eff`): Larger batch sizes lead to larger matrix multiplications, which better amortize kernel launch overhead and memory access patterns:

```
batch_eff = min(1.0, 0.55 + 0.45 × min(1.0, batch_size / 8))
```

Batch sizes ≥8 achieve full utilization; batch=1 drops to ~60%.

**Precision factor** (`mpFactor`): FP16/BF16 tensor cores are ~2× faster than FP32 CUDA cores:
- Mixed precision (FP16): 1.0
- FP32: 0.52

### 2.4 GPU Peak TFLOPS

Values used (BF16/FP16 dense tensor core, no structured sparsity):

| GPU | Peak TFLOPS | Source |
|-----|-------------|--------|
| A100 40/80GB | 312 | NVIDIA A100 datasheet |
| H100 80GB | 990 | NVIDIA H100 datasheet (FP16 TC) |
| H200 141GB | 990 | Same compute die as H100 |
| B200 192GB | 2,250 | NVIDIA B200 specs (estimated FP16 dense) |

### 2.5 Iteration Time

```
time_per_iter = Total_FLOPs / (HFU × peak_TFLOPS × num_GPUs × 1e12)
```

The global batch size used in FLOPs calculation is paradigm-dependent:
- DDP/FSDP: `batch_size × N` (data-parallel)
- PP: `batch_size × N` (pipeline fills with N micro-batches)
- TP: `batch_size` (all GPUs process the same batch)
- 3D: `batch_size × dp_degree`

---

## 3. Training Simulation

### 3.1 Phase Model

Each training iteration is divided into four phases with proportional durations:

| Phase | Tick range | Duration % | What happens |
|-------|-----------|-----------|--------------|
| Forward | 0–24 | 25% | Activations accumulate layer by layer |
| Backward | 25–54 | 30% | Gradients computed, activations freed |
| Communication | 55–77 | 23% | AllReduce / AllGather / ReduceScatter |
| Optimizer | 78–99 | 22% | Adam update, state written |

The backward pass is longer than forward (gradient computation involves both weight gradients and activation gradients per layer).

### 3.2 Memory During Simulation

Memory fluctuates during each iteration based on phase:

- **Forward**: 55% → 100% of steady-state perGPU (activations accumulating)
- **Backward**: 100% → 118% (peak — gradients coexist with remaining activations)
- **Communication**: 108% → 88% (activations being freed, gradients being communicated)
- **Optimizer**: 92% → 84% (optimizer step, then state cleanup)

The 118% peak during backward is realistic — this transient spike is why "barely fits" configurations can still OOM in practice.

### 3.3 Pipeline Parallelism Stagger

For PP, each GPU's phase is delayed proportionally to its stage index:

```
effective_tick = tick - (gpu_index / num_GPUs) × 30
```

If `effective_tick < 0`, the GPU is idle (pipeline bubble). This creates the characteristic stagger where later stages lag behind earlier ones, and the visible temperature/memory differences between stages at any given moment.

### 3.4 Temperature Model

GPU temperature is modeled as a function of computational load:

| Phase | Base temp (°C) | Range | Rationale |
|-------|---------------|-------|-----------|
| Forward | 66 → 80 | Active compute, warming | Sustained matmul |
| Backward | 74 → 86 | Peak compute | Backward is heavier than forward |
| Communication | 58 → 53 | Cooling | GPU waiting for network I/O |
| Optimizer | 64 → 74 | Moderate compute | Element-wise operations |
| Idle (PP bubble) | ~38 | Cool | No work being done |

A sinusoidal noise of ±2.5°C is added per GPU: `sin(gpuIdx × 7.3 + iterSeed × 3.1 + tick × 0.17) × 2.5`. This creates realistic per-GPU variation (different silicon, different airflow positions in the chassis).

### 3.5 3D Parallel Decomposition

The auto-decomposition heuristic for 3D parallelism:

```
tp_degree = min(8, num_GPUs)           // Fill one node with TP first
pp_degree = min(remaining, ceil(L/8))   // Use PP for deep models
dp_degree = num_GPUs / (tp × pp)        // Remaining dimension is DP
```

This follows the standard practice: TP within a node (needs NVLink), PP across node groups (only P2P activations cross the boundary), DP for throughput scaling.

---

## 4. Known Simplifications

1. **Activation memory** uses a simplified factor rather than per-architecture tensor counting. Real activation memory depends on attention mechanism (MHA vs GQA vs MQA), MLP expansion ratio (typically 4× or 8/3×), and whether intermediate attention scores are stored.

2. **FSDP temporary buffers** are not modeled. During all-gather, FSDP temporarily holds the full layer's parameters in memory (a transient spike of ~1 layer's worth of params). This is freed immediately after the layer's forward/backward.

3. **Communication overlap** is captured as a single efficiency factor rather than modeling the actual overlap between computation and communication. In practice, DDP overlaps AllReduce with backward via bucketed gradient communication, and FSDP can prefetch the next layer's all-gather during the current layer's compute.

4. **Sequence parallelism** (an extension of TP that also shards activations along the sequence dimension) is not modeled. With SP, TP's activation memory would also be divided by the TP degree.

5. **Expert parallelism** for Mixture-of-Experts models is not modeled. MoE models like Mixtral have different memory and communication patterns.

6. **Gradient accumulation** is not modeled as a separate feature. In practice, gradient accumulation allows using a larger effective batch size without proportionally more activation memory.

7. **The HFU model is empirical, not analytical.** The coefficients (0.035 for DDP comm degradation, etc.) are fitted to approximate published results from papers like PaLM, LLaMA, and Megatron-LM, not derived from first principles.

8. **Gemini Ultra architecture** is estimated. Google has not published full architecture details. The 540B parameter count and architecture dimensions are community estimates.
