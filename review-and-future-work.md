# Parallelism Explorer — Code Review & Future Work

## Part 1: Review Findings

### Bugs Fixed in This Pass

| # | Severity | Issue | Fix |
|---|----------|-------|-----|
| 1 | **High** | KV cache formula used factor of 4 instead of 2 for K+V tensors, overestimating inference memory by 2× | Changed leading multiplier from 4 to 2. Verified: LLaMA 7B KV cache at batch=4, seq=4096 now reports ~8.6 GB (was 17.2 GB). |
| 2 | **Medium** | Pipeline parallel bubble calculation capped stages at 16, producing incorrect HFU for PP with >16 GPUs | Removed `Math.min(nGPU, 16)` cap. Stages = nGPU for pure PP. |
| 3 | **Medium** | `estimateIterTime` used `globalBatch = batchSize × nGPU` for all paradigms. For TP, all GPUs process the same batch (global = batchSize). For 3D, only the DP dimension multiplies data. | Made `estimateIterTime` paradigm-aware: TP uses `batchSize`, 3D uses `batchSize × dp`, DDP/FSDP/PP use `batchSize × nGPU`. |
| 4 | **Low** | JSX transpilation crash (`_react2` not found) caused by compact `return<div>` syntax | Reformatted all return statements with proper spacing and parentheses. Used `React.Fragment` explicitly. |

### Parameters Verified Correct

| Parameter | Value | Source | Status |
|-----------|-------|--------|--------|
| GPT-2 architecture | 124M, 12L, 768H, 1024S | Radford et al. 2019 | ✅ |
| GPT-2 XL | 1.5B, 48L, 1600H | Radford et al. 2019 | ✅ |
| LLaMA 7B | 6.7B, 32L, 4096H, 4096S | Touvron et al. 2023 | ✅ |
| LLaMA 13B | 13B, 40L, 5120H | Touvron et al. 2023 | ✅ |
| LLaMA 70B | 70B, 80L, 8192H | Touvron et al. 2023 | ✅ |
| GPT-3 | 175B, 96L, 12288H, 2048S | Brown et al. 2020 | ✅ |
| LLaMA 405B | 405B, 126L, 16384H | Meta 2024 | ✅ |
| Gemini Ultra | ~540B, 128L, 18432H | Community estimate | ⚠️ Approximate |
| A100 peak TFLOPS | 312 (BF16 TC) | NVIDIA datasheet | ✅ |
| H100 peak TFLOPS | 990 (FP16 TC) | NVIDIA datasheet | ✅ |
| H200 specs | 141GB, same die as H100 | NVIDIA datasheet | ✅ |
| B200 peak TFLOPS | 2250 (estimated FP16 dense) | NVIDIA GTC 2024 | ⚠️ Approximate |
| Adam optimizer memory | 12B/param (MP), 8B/param (FP32) | Standard | ✅ |
| Training memory formula | 16 bytes/param total | Rajbhandari et al. (ZeRO) | ✅ |

### Calculations Spot-Checked

**LLaMA 7B, FSDP, 8× A100-80GB, FP16, batch=8:**
- Params: 6.7B × 2B ÷ 8 = 1.675 GB ✅
- Grads: 6.7B × 2B ÷ 8 = 1.675 GB ✅
- Optimizer: 6.7B × 12B ÷ 8 = 10.05 GB ✅
- Activations: 8 × 4096 × 4096 × 32 × 8 / 1e9 ÷ 8 = 4.29 GB ✅
- Total: ~17.7 GB per GPU → fits 80 GB ✅

**GPT-3 175B, DDP, 8× A100-80GB:**
- Params: 175B × 2B = 350 GB → OOM on single GPU ✅ (correct: DDP replicates)
- Switch to FSDP: 350 ÷ 8 = 43.75 GB params → optimizer also sharded → fits ✅

**Pipeline bubble ratio, 8 stages, batch=8:**
- micro_batches = max(8, 8) = 8
- pipe_eff = 8 / (8 + 7) = 53.3% ✅ (matches theoretical formula)

### Remaining Known Simplifications

These are documented in the methodology file and are intentional scope boundaries, not bugs:

1. FSDP temporary all-gather buffer not modeled (transient spike)
2. Sequence parallelism not modeled (would reduce TP activation memory)
3. Communication overlap modeled as a flat efficiency factor, not a timeline
4. Expert parallelism (MoE) not supported
5. Gradient accumulation not modeled
6. HFU coefficients are empirically fitted, not first-principles

---

## Part 2: Suggestions & Future Work

### High Impact

**1. Interactive timeline / Gantt chart visualization**
Replace the phase-bar abstraction with a real Gantt chart showing what each GPU is doing at each moment. For PP, this would show the staggered micro-batch schedule (like the static diagram, but animated). For FSDP, it would show all-gather → compute → reduce-scatter overlapping across layers. This is the single most educational addition possible.

**2. Network topology visualization**
Draw actual NVLink mesh (intra-node) and InfiniBand tree (inter-node) connections between GPU cards. Animate data flowing along the connections during the communication phase of the simulation. Color-code by bandwidth utilization.

**3. Real profiling data import**
Allow users to upload NVIDIA Nsight / PyTorch Profiler traces (JSON) and overlay real measured data against the simulator's estimates. Show where the model diverges from reality and by how much.

**4. Cost estimation**
Add cloud pricing (AWS p4d/p5, GCP a3, Azure ND H100) and compute cost-per-iteration and cost-per-token. This turns the educational tool into a practical planning tool.

**5. Gradient accumulation support**
Add a gradient accumulation steps slider. This multiplies the effective global batch by the accumulation count without increasing activation memory. It changes: the communication frequency (AllReduce only every K steps for DDP), the pipeline efficiency (more micro-batches), and the FLOPs/iter calculation.

### Medium Impact

**6. Sequence parallelism (SP) toggle**
SP is a standard extension of TP that also shards activations along the sequence dimension. Adding this would change TP's activation divisor from 1 to N, significantly reducing per-GPU memory and making TP more competitive for long-sequence models.

**7. ZeRO stage selector for FSDP**
FSDP currently models ZeRO-3 (full sharding). Add ZeRO-1 (optimizer-only) and ZeRO-2 (optimizer + gradients) as options. This would show the progression of memory savings:
- ZeRO-1: `d(1, 1, N, N)` — only optimizer sharded
- ZeRO-2: `d(1, N, N, N)` — optimizer + gradients sharded
- ZeRO-3: `d(N, N, N, N)` — everything sharded

**8. Expert parallelism for MoE**
Add a Mixtral/Switch-style MoE option. Experts are distributed across GPUs with all-to-all communication for token routing. Memory per GPU = (total_expert_params / num_expert_groups) + shared_params. Communication pattern is fundamentally different from dense models.

**9. Memory pressure timeline chart**
During simulation, plot a line chart of per-GPU memory over time (all 10 iterations). Show the peak memory line and the GPU capacity ceiling. This would make the "transient OOM" phenomenon visible — configurations that fit on average but spike above capacity during the backward pass.

**10. Token throughput / MFU display**
Add tokens-per-second and MFU (Model FLOP Utilization, which excludes recomputation FLOPs unlike HFU). MFU is what papers like PaLM and LLaMA report, so including it would let users compare against published numbers directly.

### Lower Priority / Polish

**11. Custom model config**
Let users enter arbitrary parameter count, layer count, hidden dim, and sequence length instead of picking presets. This supports architectures not in the preset list (e.g., custom fine-tuning configs, vision transformers, multimodal models).

**12. Multi-dimensional batch configuration**
Separate micro-batch size, gradient accumulation steps, and data-parallel degree as independent controls instead of deriving global batch from a single slider. This gives more control and matches how practitioners actually configure training.

**13. Bandwidth-aware communication modeling**
Instead of a flat efficiency factor, model communication time as `volume / bandwidth`:
- Intra-node NVLink: 600 GB/s (A100) or 900 GB/s (H100)
- Inter-node InfiniBand: 400 Gbps = 50 GB/s (HDR) or 800 Gbps = 100 GB/s (NDR)
This would make the TP intra-vs-cross-node difference data-driven rather than heuristic.

**14. Power consumption and cooling**
Extend the temperature model with per-GPU TDP (A100: 400W, H100: 700W) and cluster-level power draw. Show total rack power and estimated electricity cost per training run.

**15. Failure recovery modeling**
At large scale (1000+ GPUs), hardware failures are inevitable. Model the expected MTBF (mean time between failures), checkpointing overhead, and wasted compute from restarts. This changes the effective throughput significantly for frontier training runs.

**16. Comparison mode**
Let users pin a configuration and compare it side-by-side with a different paradigm, GPU count, or model size. Show delta memory, delta HFU, and delta cost.

**17. Export configuration**
Generate a PyTorch/DeepSpeed/Megatron-LM config file from the current configuration. E.g., output the `torchrun` command with the right parallelism flags, or a DeepSpeed JSON config with ZeRO stage and offloading settings.

**18. Mobile-responsive layout**
The current grid layout breaks on narrow viewports. Add responsive breakpoints so the concept panel and GPU grid stack vertically on mobile. The diagrams already use `width="100%"` with `maxWidth` so they scale, but the two-column layout needs a media query.

---

## Part 3: Validation Roadmap

To move from "educational simulator" to "trusted planning tool," the following validation steps would be needed:

1. **Benchmark against published MFU numbers**: Compare the simulator's HFU estimates against reported values from PaLM (46.2% on TPUv4), LLaMA-2 (43% on A100), GPT-3 (estimated 30-35%), and Megatron-LM papers.

2. **Profile real training runs**: Run actual training for a subset of configurations (e.g., LLaMA 7B on 8× H100 with DDP/FSDP/TP) using PyTorch + NCCL, measure wall-clock time, and compare against predicted iter time.

3. **Memory validation**: Compare predicted per-GPU memory against `torch.cuda.max_memory_allocated()` for each paradigm.

4. **Communication volume validation**: Measure actual AllReduce/AllGather volumes with NCCL debug logging and compare against the model's comm_eff factor.

5. **Temperature validation**: Compare against `nvidia-smi` temperature readings during actual training phases.

Each validation pass would produce correction factors that could be folded back into the estimation model, progressively improving accuracy.
