import React, { useState, useMemo, useEffect, useRef } from "react";

const T = {
  bg: "#F4F6FB", surface: "#FFFFFF", card: "#FFFFFF",
  border: "#DDE3ED", borderSoft: "#EDF1F7",
  text: "#1A2138", mid: "#3B4562", muted: "#6B7A94", subtle: "#9AA5B8",
  ddp: "#4E6AD4", fsdp: "#7556D0", pp: "#C67E28", tp: "#24956E", threeD: "#B84A6C",
  params: "#6E9CF0", grads: "#E8A052", optim: "#A484E0", act: "#5CC48E", kv: "#D8C048",
  free: "#ECF0F6", ok: "#2E9460", danger: "#C44040", warn: "#C49030",
  shadow: "0 1px 4px rgba(30,40,70,0.05)",
};
const mono = { fontFamily: "'SF Mono','Cascadia Code',Consolas,monospace" };

const MODELS = [
  { id: "gpt2", name: "GPT-2", p: 0.124, L: 12, H: 768, S: 1024 },
  { id: "gpt2xl", name: "GPT-2 XL", p: 1.5, L: 48, H: 1600, S: 1024 },
  { id: "llama7", name: "LLaMA 7B", p: 6.7, L: 32, H: 4096, S: 4096 },
  { id: "llama13", name: "LLaMA 13B", p: 13, L: 40, H: 5120, S: 4096 },
  { id: "llama70", name: "LLaMA 70B", p: 70, L: 80, H: 8192, S: 4096 },
  { id: "gpt3", name: "GPT-3 175B", p: 175, L: 96, H: 12288, S: 2048 },
  { id: "l405", name: "LLaMA 405B", p: 405, L: 126, H: 16384, S: 8192 },
  { id: "gemini", name: "Gemini Ultra", p: 540, L: 128, H: 18432, S: 8192 },
];
const GPUS = [
  { id: "a100-40", name: "A100 40GB", mem: 40, peak: 312 },
  { id: "a100-80", name: "A100 80GB", mem: 80, peak: 312 },
  { id: "h100", name: "H100 80GB", mem: 80, peak: 990 },
  { id: "h200", name: "H200 141GB", mem: 141, peak: 990 },
  { id: "b200", name: "B200 192GB", mem: 192, peak: 2250 },
];
const PARADIGMS = [
  { id: "ddp", name: "DDP", full: "Distributed Data Parallel", c: T.ddp },
  { id: "fsdp", name: "FSDP", full: "Fully Sharded Data Parallel", c: T.fsdp },
  { id: "pp", name: "Pipeline", full: "Pipeline Parallelism", c: T.pp },
  { id: "tp", name: "Tensor", full: "Tensor Parallelism", c: T.tp },
  { id: "3d", name: "3D", full: "3D Parallel (TP+PP+DP)", c: T.threeD },
];
const GPU_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128];
const CONCEPTS = {
  ddp: { title: "Distributed Data Parallel", pts: ["Every GPU holds a complete copy of the model.", "Training data is split — each GPU processes a different mini-batch.", "Gradients are averaged via AllReduce after the backward pass.", "Near-linear throughput. Use when the model fits on one GPU."], comm: "AllReduce (gradients, once per step)" },
  fsdp: { title: "Fully Sharded Data Parallel", pts: ["Params, gradients, and optimizer states are sharded — each GPU holds 1/N.", "Before each layer's forward, All-Gather reconstructs the full parameters.", "After backward, Reduce-Scatter distributes gradient shards back.", "Memory drops from O(model) to O(model/N) per GPU."], comm: "All-Gather (fwd) + Reduce-Scatter (bwd)" },
  pp: { title: "Pipeline Parallelism", pts: ["The model is split by layers into sequential stages across GPUs.", "Each stage passes output activations to the next — like an assembly line.", "Naïve pipelining causes idle 'bubbles'. Micro-batching reduces them.", "Best for very deep models with limited inter-GPU bandwidth."], comm: "Point-to-point (activations between stages)" },
  tp: { title: "Tensor Parallelism", pts: ["Weight matrices split column/row-wise across GPUs within each layer.", "Each GPU computes a partial product, AllReduce sums results.", "Every layer requires synchronization — needs fast NVLink.", "Powers Megatron-LM style training for wide transformer layers."], comm: "AllReduce per layer (intra-node NVLink)" },
  "3d": { title: "3D Parallelism", pts: ["Combines TP within a node, PP across node groups, DP/FSDP across replicas.", "Auto-decomposes: e.g. 64 GPUs → TP=8 × PP=4 × DP=2.", "Each axis targets a bottleneck: TP for width, PP for depth, DP for throughput.", "Standard recipe for frontier models (LLaMA 405B, Gemini)."], comm: "AllReduce (TP) + P2P (PP) + AllGather (DP)" },
};

/* ═══════════════ CALCULATIONS ═══════════════ */
function calcTrain(m, par, N, mp, ckpt, bs) {
  const pGB = m.p * (mp ? 2 : 4);
  const gGB = m.p * (mp ? 2 : 4);
  const oGB = m.p * (mp ? 12 : 8);
  const aGB = ckpt
    ? Math.max(0.02, bs * m.S * m.H * Math.sqrt(m.L) * 8 / 1e9)
    : Math.max(0.05, bs * m.S * m.H * m.L * 8 / 1e9);
  const d = (pd, gd, od, ad) => ({
    params: pGB / pd, grads: gGB / gd, optim: oGB / od, act: aGB / ad,
    totP: pGB, totG: gGB, totO: oGB, totA: aGB,
  });
  if (par === "ddp") return d(1, 1, 1, N);
  if (par === "fsdp") return d(N, N, N, N);
  if (par === "pp") return d(N, N, N, N);
  if (par === "tp") return d(N, N, N, 1);
  if (par === "3d") {
    const tp = Math.min(8, N);
    const rest = N / tp;
    const pp = rest > 1 ? Math.min(rest, Math.ceil(m.L / 8)) : 1;
    const dp = Math.max(1, N / (tp * pp));
    return { ...d(tp * pp, tp * pp, tp * pp * dp, dp), tp, pp, dp };
  }
  return d(1, 1, 1, N);
}

function calcInfer(m, par, N) {
  const pGB = m.p * 2;
  // KV cache = 2 (K+V) × layers × hidden_dim × seq_len × batch × 2 bytes (fp16)
  const kvGB = Math.max(0.01, 2 * m.L * m.H * m.S * 4 * 2 / 1e9);
  if (par === "ddp") return { params: pGB, kv: kvGB };
  if (par === "fsdp") return { params: pGB / N, kv: kvGB };
  if (par === "pp") return { params: pGB / N, kv: kvGB / N };
  if (par === "tp") return { params: pGB / N, kv: kvGB / N };
  if (par === "3d") {
    const tp = Math.min(8, N);
    const pp = Math.max(1, Math.floor(N / tp));
    return { params: pGB / (tp * pp), kv: kvGB / pp, tp, pp };
  }
  return { params: pGB, kv: kvGB };
}

/* ═══════════════ SIMULATION ═══════════════ */
function tempColor(t) {
  if (t < 45) return "#4A96D8";
  if (t < 55) return "#40A89A";
  if (t < 65) return "#6BBF6A";
  if (t < 75) return "#D8B840";
  if (t < 82) return "#D88040";
  return "#C84040";
}

function getSimGPU(gpuIdx, tick, par, nGPU, perGPU, iterSeed) {
  let effTick = tick;
  if (par === "pp") {
    const delay = (gpuIdx / Math.max(nGPU, 1)) * 30;
    effTick = tick - delay;
    if (effTick < 0) {
      return { phase: "idle", mem: perGPU * 0.35, temp: Math.round(38 + Math.sin(gpuIdx * 5 + iterSeed) * 2) };
    }
  }
  let phase, prog;
  if (effTick < 25) { phase = "forward"; prog = effTick / 25; }
  else if (effTick < 55) { phase = "backward"; prog = (effTick - 25) / 30; }
  else if (effTick < 78) { phase = "comm"; prog = (effTick - 55) / 23; }
  else { phase = "optim"; prog = (effTick - 78) / 22; }
  const n = Math.sin(gpuIdx * 7.3 + iterSeed * 3.1 + tick * 0.17) * 2.5;
  let mem, temp;
  if (phase === "forward") { mem = perGPU * (0.55 + 0.45 * prog); temp = 66 + 14 * prog + n; }
  else if (phase === "backward") { mem = perGPU * (1.0 + 0.18 * prog); temp = 74 + 12 * prog + n; }
  else if (phase === "comm") { mem = perGPU * (1.08 - 0.2 * prog); temp = 58 - 5 * prog + n; }
  else { mem = perGPU * (0.92 - 0.08 * prog); temp = 64 + 10 * prog + n; }
  return { phase, mem: Math.max(0, mem), temp: Math.round(Math.max(30, Math.min(95, temp))) };
}

/* ═══════════════ HFU ESTIMATION ═══════════════ */
function estimateHFU(model, par, nGPU, batchSize, mp, ckpt) {
  // 1. Compute efficiency: how well matmuls occupy tensor cores
  //    Larger hidden → larger tiles → better occupancy
  const computeEff = Math.min(0.68, 0.30 + (model.H / 24000) * 0.38);

  // 2. Communication overhead by paradigm
  const logN = Math.max(0, Math.log2(nGPU));
  let commEff;
  if (par === "ddp") {
    // AllReduce once per step, overlaps with backward
    commEff = Math.max(0.65, 1.0 - 0.035 * logN);
  } else if (par === "fsdp") {
    // AllGather + ReduceScatter per layer, partial overlap
    commEff = Math.max(0.50, 0.93 - 0.055 * logN);
  } else if (par === "pp") {
    // Minimal comm (P2P activations), but bubbles handled separately
    commEff = Math.max(0.80, 0.98 - 0.015 * logN);
  } else if (par === "tp") {
    // AllReduce per layer — heavy but fast over NVLink intra-node
    commEff = nGPU <= 8
      ? Math.max(0.72, 0.95 - 0.03 * logN)   // intra-node NVLink
      : Math.max(0.45, 0.85 - 0.06 * logN);   // cross-node penalty
  } else {
    // 3D: product of TP, PP comm, DP
    const tp = Math.min(8, nGPU);
    const pp = Math.max(1, Math.min(nGPU / tp, Math.ceil(model.L / 8)));
    const dp = Math.max(1, nGPU / (tp * pp));
    const tpC = Math.max(0.75, 0.95 - 0.03 * Math.log2(tp));
    const dpC = dp > 1 ? Math.max(0.80, 0.96 - 0.03 * Math.log2(dp)) : 1.0;
    commEff = tpC * dpC;
  }

  // 3. Pipeline bubble efficiency (only matters for PP and 3D)
  let pipeEff = 1.0;
  if (par === "pp") {
    const stages = nGPU;
    const microBatches = Math.max(stages, batchSize);
    pipeEff = Math.max(0.25, microBatches / (microBatches + stages - 1));
  } else if (par === "3d") {
    const tp = Math.min(8, nGPU);
    const pp = Math.max(1, Math.min(nGPU / tp, Math.ceil(model.L / 8)));
    if (pp > 1) {
      const microBatches = Math.max(pp, batchSize);
      pipeEff = Math.max(0.40, microBatches / (microBatches + pp - 1));
    }
  }

  // 4. Batch utilization: larger batch → better GEMM efficiency
  const batchEff = Math.min(1.0, 0.55 + 0.45 * Math.min(1.0, batchSize / 8));

  // 5. Mixed precision: FP16 tensor cores are ~2× faster
  const mpEff = mp ? 1.0 : 0.52;

  const hfu = computeEff * commEff * pipeEff * batchEff * mpEff;
  return {
    hfu: Math.min(0.62, Math.max(0.03, hfu)),
    compute: computeEff,
    comm: commEff,
    pipe: pipeEff,
    batch: batchEff,
    mpFactor: mpEff,
  };
}

function estimateIterTime(model, par, nGPU, batchSize, hfu, peakTFLOPS, ckpt) {
  // FLOPs per iteration: 6*P*B_global*S (fwd+bwd), 8 if recompute
  const flopsMultiplier = ckpt ? 8 : 6;
  // Global batch depends on paradigm:
  // DDP/FSDP: each GPU gets its own data shard → global = batchSize × nGPU
  // PP: pipeline fills with micro-batches → global = batchSize × nGPU (filling pipeline)
  // TP: all GPUs compute the same batch → global = batchSize
  // 3D: only DP dimension multiplies data → global = batchSize × dp
  let globalBatch;
  if (par === "tp") {
    globalBatch = batchSize;
  } else if (par === "3d") {
    const tp = Math.min(8, nGPU);
    const pp = Math.max(1, Math.min(nGPU / tp, Math.ceil(model.L / 8)));
    const dp = Math.max(1, nGPU / (tp * pp));
    globalBatch = batchSize * dp;
  } else {
    globalBatch = batchSize * nGPU;
  }
  const totalFLOPs = flopsMultiplier * model.p * 1e9 * globalBatch * model.S;
  const achievedTFLOPS = hfu * peakTFLOPS * nGPU;
  if (achievedTFLOPS <= 0) return 0;
  return totalFLOPs / (achievedTFLOPS * 1e12); // seconds
}

/* ═══════════════ FIXED SVG DIAGRAMS ═══════════════ */
function DiagramDDP() {
  const W = 460, gw = 90;
  const xs = [20, 135, 250, 365];
  const cols = ["#CFDDFA", "#FADDC8", "#DACBFA", "#C5ECD6"];
  const strk = [T.ddp, "#C4822A", "#7E5AC0", "#24956E"];
  return (
    <svg viewBox={`0 0 ${W} 210`} width="100%" style={{ maxWidth: 500 }}>
      {xs.map((x, i) => (
        <g key={"b" + i}>
          <rect x={x} y={8} width={gw} height={24} rx={4} fill={cols[i]} stroke={strk[i]} strokeWidth={1} />
          <text x={x + 45} y={24} textAnchor="middle" fill={strk[i]} style={{ fontSize: 9, ...mono, fontWeight: 600 }}>Batch {i}</text>
        </g>
      ))}
      {xs.map((x, i) => (
        <line key={"a" + i} x1={x + 45} y1={34} x2={x + 45} y2={50} stroke={strk[i]} strokeWidth={1.2} />
      ))}
      {xs.map((x, i) => (
        <g key={"g" + i}>
          <rect x={x} y={52} width={gw} height={50} rx={6} fill={T.surface} stroke={T.ddp} strokeWidth={1.3} />
          <text x={x + 45} y={72} textAnchor="middle" fill={T.ddp} style={{ fontSize: 10, ...mono, fontWeight: 600 }}>GPU {i}</text>
          <text x={x + 45} y={90} textAnchor="middle" fill={T.muted} style={{ fontSize: 8, ...mono }}>Full Model</text>
        </g>
      ))}
      {[0, 1, 2].map(i => (
        <g key={"s" + i}>
          <line x1={xs[i] + gw + 2} y1={77} x2={xs[i + 1] - 2} y2={77} stroke={T.ddp} strokeWidth={1} strokeDasharray="4,3" />
          <polygon points={`${xs[i + 1] - 2},77 ${xs[i + 1] - 8},74 ${xs[i + 1] - 8},80`} fill={T.ddp} />
        </g>
      ))}
      <path d={`M${xs[0] + 45},104 Q${xs[0] - 12},148 ${W / 2},152 Q${xs[3] + gw + 12},148 ${xs[3] + 45},104`}
        fill="none" stroke={T.ddp} strokeWidth={1.2} strokeDasharray="5,3" opacity={0.5} />
      <rect x={W / 2 - 74} y={138} width={148} height={24} rx={5} fill="#EDF0FA" stroke={T.ddp} strokeWidth={1} />
      <text x={W / 2} y={154} textAnchor="middle" fill={T.ddp} style={{ fontSize: 10, ...mono, fontWeight: 600 }}>AllReduce ∇ gradients</text>
      <text x={W / 2} y={188} textAnchor="middle" fill={T.muted} style={{ fontSize: 8, ...mono }}>Same weights → different data → averaged gradients</text>
    </svg>
  );
}

function FSDPPhaseRow({ y, label, color, shardColors, shardStrokes, gpuLabels, xs, gw }) {
  return (
    <g>
      <text x={230} y={y} textAnchor="middle" fill={color} style={{ fontSize: 9, ...mono, fontWeight: 600 }}>{label}</text>
      {xs.map((x, i) => (
        <g key={i}>
          <rect x={x} y={y + 8} width={gw} height={34} rx={5} fill={shardColors[i]} stroke={shardStrokes[i]} strokeWidth={1} />
          <text x={x + gw / 2} y={y + 22} textAnchor="middle" fill={T.text} style={{ fontSize: 9, ...mono, fontWeight: 600 }}>GPU {i}</text>
          <text x={x + gw / 2} y={y + 36} textAnchor="middle" fill={T.muted} style={{ fontSize: 7, ...mono }}>{gpuLabels(i)}</text>
        </g>
      ))}
    </g>
  );
}

function DiagramFSDP() {
  const W = 460, gw = 85;
  const xs = [18, 128, 238, 348];
  const sc = ["#CFDDFA", "#DACBFA", "#FADDC8", "#C5ECD6"];
  const sk = [T.ddp, T.fsdp, T.pp, T.tp];
  const allFill = ["#EDEBFA", "#EDEBFA", "#EDEBFA", "#EDEBFA"];
  return (
    <svg viewBox={`0 0 ${W} 265`} width="100%" style={{ maxWidth: 500 }}>
      <FSDPPhaseRow y={10} label="① Idle — each GPU holds 1/N" color={T.muted}
        shardColors={sc} shardStrokes={sk} gpuLabels={i => `Shard ${i}`} xs={xs} gw={gw} />
      <text x={W / 2} y={66} textAnchor="middle" fill={T.fsdp} style={{ fontSize: 14 }}>↓</text>
      <FSDPPhaseRow y={74} label="② All-Gather → full layer on each GPU" color={T.fsdp}
        shardColors={allFill} shardStrokes={[T.fsdp, T.fsdp, T.fsdp, T.fsdp]} gpuLabels={() => "Full Layer"} xs={xs} gw={gw} />
      <text x={W / 2} y={132} textAnchor="middle" fill={T.mid} style={{ fontSize: 14 }}>↓</text>
      <text x={W / 2} y={148} textAnchor="middle" fill={T.mid} style={{ fontSize: 9, ...mono, fontWeight: 600 }}>③ Forward / Backward pass</text>
      <text x={W / 2} y={166} textAnchor="middle" fill={T.mid} style={{ fontSize: 14 }}>↓</text>
      <FSDPPhaseRow y={174} label="④ Reduce-Scatter → keep own grad shard" color={T.pp}
        shardColors={sc} shardStrokes={sk} gpuLabels={i => `∇ Shard ${i}`} xs={xs} gw={gw} />
      <text x={W / 2} y={236} textAnchor="middle" fill={T.muted} style={{ fontSize: 8, ...mono }}>Temporary full-layer buffer freed after each layer</text>
    </svg>
  );
}

function DiagramPP() {
  const W = 490, gw = 92;
  const xs = [16, 132, 248, 364];
  const cols = ["#6E9CF0", "#A484E0", "#E8A052", "#5CC48E"];
  const bg = ["#CFDDFA", "#DACBFA", "#FADDC8", "#C5ECD6"];
  const stages = ["L0–7", "L8–15", "L16–23", "L24–31"];
  return (
    <svg viewBox={`0 0 ${W} 250`} width="100%" style={{ maxWidth: 530 }}>
      <text x={W / 2} y={14} textAnchor="middle" fill={T.muted} style={{ fontSize: 8, ...mono }}>Model split by layers into sequential stages</text>
      {xs.map((x, i) => (
        <g key={"st" + i}>
          <rect x={x} y={22} width={gw} height={42} rx={6} fill={bg[i]} stroke={T.pp} strokeWidth={1.2} />
          <text x={x + 46} y={40} textAnchor="middle" fill={T.pp} style={{ fontSize: 10, ...mono, fontWeight: 600 }}>Stage {i}</text>
          <text x={x + 46} y={55} textAnchor="middle" fill={T.muted} style={{ fontSize: 8, ...mono }}>{stages[i]}</text>
        </g>
      ))}
      {[0, 1, 2].map(i => (
        <g key={"ar" + i}>
          <line x1={xs[i] + gw + 3} y1={43} x2={xs[i + 1] - 3} y2={43} stroke={T.pp} strokeWidth={1.3} />
          <polygon points={`${xs[i + 1] - 3},43 ${xs[i + 1] - 9},40 ${xs[i + 1] - 9},46`} fill={T.pp} />
          <text x={(xs[i] + gw + xs[i + 1]) / 2} y={37} textAnchor="middle" fill={T.pp} style={{ fontSize: 7, ...mono }}>act</text>
        </g>
      ))}
      <text x={14} y={88} fill={T.mid} style={{ fontSize: 9, ...mono, fontWeight: 600 }}>Schedule (time →)</text>
      {[0, 1, 2, 3].map(s => {
        const y = 98 + s * 32;
        const cw = 50;
        return (
          <g key={"sch" + s}>
            <text x={8} y={y + 17} fill={T.muted} style={{ fontSize: 9, ...mono }}>S{s}</text>
            {s > 0 && Array.from({ length: s }, (_, b) => (
              <g key={"bub" + b}>
                <rect x={28 + b * cw} y={y} width={cw - 4} height={26} rx={4} fill={T.danger} opacity={0.08} stroke={T.danger} strokeWidth={0.6} strokeDasharray="3,2" />
                <text x={28 + b * cw + (cw - 4) / 2} y={y + 16} textAnchor="middle" fill={T.danger} style={{ fontSize: 8, ...mono }} opacity={0.5}>idle</text>
              </g>
            ))}
            <rect x={28 + s * cw} y={y} width={cw - 4} height={26} rx={4} fill={cols[s]} opacity={0.75} />
            <text x={28 + s * cw + (cw - 4) / 2} y={y + 16} textAnchor="middle" fill="#fff" style={{ fontSize: 9, ...mono, fontWeight: 600 }}>Fwd</text>
            <rect x={28 + 4 * cw + s * cw} y={y} width={cw - 4} height={26} rx={4} fill={cols[s]} opacity={0.3} stroke={cols[s]} strokeWidth={1} />
            <text x={28 + 4 * cw + s * cw + (cw - 4) / 2} y={y + 16} textAnchor="middle" fill={T.text} style={{ fontSize: 8, ...mono }}>Bwd</text>
          </g>
        );
      })}
      <text x={28} y={238} fill={T.danger} style={{ fontSize: 8, ...mono }}>■ idle = pipeline bubble</text>
      <text x={190} y={238} fill={T.muted} style={{ fontSize: 8, ...mono }}>Micro-batching overlaps work to fill gaps</text>
    </svg>
  );
}

function DiagramTP() {
  const W = 440, gw = 86, gap = 20, sx = 18;
  const cols = ["#CFDDFA", "#DACBFA", "#FADDC8", "#C5ECD6"];
  const sk = [T.ddp, T.fsdp, T.pp, T.tp];
  return (
    <svg viewBox={`0 0 ${W} 230`} width="100%" style={{ maxWidth: 480 }}>
      <rect x={W / 2 - 44} y={4} width={88} height={28} rx={5} fill="#ECF0F6" stroke={T.border} strokeWidth={1} />
      <text x={W / 2} y={22} textAnchor="middle" fill={T.text} style={{ fontSize: 11, ...mono, fontWeight: 600 }}>Input X</text>
      {[0, 1, 2, 3].map(i => (
        <line key={"fo" + i} x1={W / 2} y1={34} x2={sx + i * (gw + gap) + gw / 2} y2={56} stroke={T.subtle} strokeWidth={1} strokeDasharray="3,2" />
      ))}
      {[0, 1, 2, 3].map(i => {
        const x = sx + i * (gw + gap);
        return (
          <g key={"gpu" + i}>
            <rect x={x} y={58} width={gw} height={50} rx={6} fill={cols[i]} stroke={sk[i]} strokeWidth={1.2} />
            <text x={x + gw / 2} y={74} textAnchor="middle" fill={T.text} style={{ fontSize: 10, ...mono, fontWeight: 600 }}>GPU {i}</text>
            <text x={x + gw / 2} y={89} textAnchor="middle" fill={sk[i]} style={{ fontSize: 8, ...mono }}>X · W[:,{i}]</text>
            <text x={x + gw / 2} y={102} textAnchor="middle" fill={T.muted} style={{ fontSize: 7, ...mono }}>partial Y{i}</text>
          </g>
        );
      })}
      {[0, 1, 2, 3].map(i => (
        <line key={"fi" + i} x1={sx + i * (gw + gap) + gw / 2} y1={110} x2={W / 2} y2={140} stroke={T.tp} strokeWidth={1} strokeDasharray="3,2" />
      ))}
      <rect x={W / 2 - 54} y={142} width={108} height={26} rx={5} fill="#D8F0E4" stroke={T.tp} strokeWidth={1.2} />
      <text x={W / 2} y={159} textAnchor="middle" fill={T.tp} style={{ fontSize: 10, ...mono, fontWeight: 700 }}>AllReduce → Y</text>
      <rect x={W / 2 - 44} y={180} width={88} height={26} rx={5} fill="#D8F0E4" stroke={T.tp} strokeWidth={1} />
      <text x={W / 2} y={197} textAnchor="middle" fill={T.text} style={{ fontSize: 11, ...mono, fontWeight: 600 }}>Y = X · W</text>
      <text x={W / 2} y={222} textAnchor="middle" fill={T.muted} style={{ fontSize: 8, ...mono }}>Repeated every layer — requires NVLink speed</text>
    </svg>
  );
}

function Diagram3D() {
  const W = 450;
  const groups = [
    { x: 8, w: 130, fill: "#E2F2E8", stroke: T.tp, title: "TP (intra-node)", labels: ["0", "1", "2", "3"], iw: 24, ic: "#B8EAD0", d1: "Split matrices", d2: "AllReduce/layer" },
    { x: 158, w: 130, fill: "#FEF3E2", stroke: T.pp, title: "PP (across nodes)", labels: ["S0", "S1", "S2", "S3"], iw: 24, ic: "#FADDC8", d1: "Split by layers", d2: "P2P activations" },
    { x: 308, w: 130, fill: "#E8EBF6", stroke: T.ddp, title: "DP / FSDP", labels: ["R0", "R1"], iw: 44, ic: "#CFDDFA", d1: "Duplicate pipeline", d2: "AllReduce grads" },
  ];
  return (
    <svg viewBox={`0 0 ${W} 195`} width="100%" style={{ maxWidth: 490 }}>
      {groups.map((g, gi) => (
        <g key={gi}>
          <rect x={g.x} y={28} width={g.w} height={100} rx={8} fill={g.fill} stroke={g.stroke} strokeWidth={1} strokeDasharray="5,3" />
          <text x={g.x + g.w / 2} y={20} textAnchor="middle" fill={g.stroke} style={{ fontSize: 9, ...mono, fontWeight: 600 }}>{g.title}</text>
          {g.labels.map((lb, j) => {
            const ix = g.x + 12 + j * (g.iw + 6);
            return (
              <g key={j}>
                <rect x={ix} y={44} width={g.iw} height={24} rx={4} fill={g.ic} stroke={g.stroke} strokeWidth={0.8} />
                <text x={ix + g.iw / 2} y={60} textAnchor="middle" fill={T.text} style={{ fontSize: 8, ...mono, fontWeight: 600 }}>{lb}</text>
              </g>
            );
          })}
          <text x={g.x + g.w / 2} y={90} textAnchor="middle" fill={T.muted} style={{ fontSize: 8, ...mono }}>{g.d1}</text>
          <text x={g.x + g.w / 2} y={104} textAnchor="middle" fill={T.muted} style={{ fontSize: 8, ...mono }}>{g.d2}</text>
        </g>
      ))}
      <line x1={140} y1={78} x2={156} y2={78} stroke={T.subtle} strokeWidth={1.2} />
      <polygon points="156,78 150,75 150,81" fill={T.subtle} />
      <line x1={290} y1={78} x2={306} y2={78} stroke={T.subtle} strokeWidth={1.2} />
      <polygon points="306,78 300,75 300,81" fill={T.subtle} />
      <rect x={W / 2 - 82} y={144} width={164} height={24} rx={5} fill="#F4F6FB" stroke={T.border} strokeWidth={1} />
      <text x={W / 2} y={160} textAnchor="middle" fill={T.text} style={{ fontSize: 10, ...mono, fontWeight: 600 }}>TP × PP × DP = Total GPUs</text>
      <text x={W / 2} y={186} textAnchor="middle" fill={T.muted} style={{ fontSize: 8, ...mono }}>e.g. 8 × 4 × 2 = 64 GPUs</text>
    </svg>
  );
}

const DIAGRAMS = { ddp: DiagramDDP, fsdp: DiagramFSDP, pp: DiagramPP, tp: DiagramTP, "3d": Diagram3D };

/* ═══════════════ UI COMPONENTS ═══════════════ */
function MemBar({ segs, total, cap, h }) {
  const height = h || 13;
  const fits = total <= cap;
  const mx = Math.max(total, cap);
  return (
    <div style={{ position: "relative" }}>
      <div style={{ display: "flex", height: height, borderRadius: 4, overflow: "hidden", background: T.free, border: `1px solid ${fits ? T.border : T.danger}50` }}>
        {segs.map((s, i) => s.v > 0.005 ? (
          <div key={i} title={`${s.l}: ${s.v.toFixed(1)} GB`}
            style={{ width: `${(s.v / mx) * 100}%`, background: s.c, transition: "width .3s", minWidth: 1, opacity: 0.82 }} />
        ) : null)}
      </div>
      {cap > 0 && (
        <div style={{ position: "absolute", top: 0, left: `${Math.min(100, (cap / mx) * 100)}%`, width: 1.5, height: height, background: fits ? T.subtle : T.danger, opacity: 0.5 }} />
      )}
    </div>
  );
}

function TempBar({ temp }) {
  const pct = Math.min(100, Math.max(0, (temp - 30) / 65 * 100));
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
      <div style={{ width: 42, height: 7, borderRadius: 4, background: "#ECF0F6", overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", borderRadius: 4, background: tempColor(temp), transition: "width .3s, background .3s" }} />
      </div>
      <span style={{ fontSize: 9, ...mono, fontWeight: 600, color: tempColor(temp), minWidth: 32 }}>{temp}°C</span>
    </div>
  );
}

function getLabel(par, i, N, m) {
  if (par === "ddp") return { top: `GPU ${i}`, bot: `Replica · batch[${i}]` };
  if (par === "fsdp") return { top: `GPU ${i}`, bot: `Shard ${i}/${N}` };
  if (par === "pp") {
    const lps = Math.ceil(m.L / N);
    const s = Math.min(i * lps, m.L);
    const e = Math.min(s + lps, m.L) - 1;
    return { top: `Stage ${i}`, bot: `L${s}–L${e}` };
  }
  if (par === "tp") return { top: `GPU ${i}`, bot: `W[:,${i}]` };
  if (par === "3d") {
    const tp = Math.min(8, N);
    const pp = Math.max(1, Math.min(N / tp, Math.ceil(m.L / 8)));
    return { top: `GPU ${i}`, bot: `TP${i % tp} PP${Math.floor(i / tp) % pp} DP${Math.floor(i / (tp * pp))}` };
  }
  return { top: `GPU ${i}`, bot: "" };
}

/* ═══════════════ MAIN APP ═══════════════ */
export default function App() {
  const [mode, setMode] = useState("training");
  const [mId, setMId] = useState("llama7");
  const [par, setPar] = useState("ddp");
  const [nGPU, setNGPU] = useState(4);
  const [gId, setGId] = useState("a100-80");
  const [mp, setMp] = useState(true);
  const [ckpt, setCkpt] = useState(false);
  const [batchSize, setBatchSize] = useState(8);
  const [simOn, setSimOn] = useState(false);
  const [simProg, setSimProg] = useState(0);
  const simRef = useRef(null);
  const TOTAL_SIM = 1000;

  const model = MODELS.find(m => m.id === mId);
  const gpu = GPUS.find(g => g.id === gId);
  const pInfo = PARADIGMS.find(p => p.id === par);
  const concept = CONCEPTS[par];
  const Diagram = DIAGRAMS[par];

  const mem = useMemo(() => mode === "training" ? calcTrain(model, par, nGPU, mp, ckpt, batchSize) : calcInfer(model, par, nGPU), [mode, model, par, nGPU, mp, ckpt, batchSize]);
  const perGPU = useMemo(() => mode === "training" ? mem.params + mem.grads + mem.optim + mem.act : mem.params + mem.kv, [mode, mem]);
  const fits = perGPU <= gpu.mem;

  const segs = useMemo(() => mode === "training"
    ? [{ l: "Params", v: mem.params, c: T.params }, { l: "Grads", v: mem.grads, c: T.grads }, { l: "Optim", v: mem.optim, c: T.optim }, { l: "Act", v: mem.act, c: T.act }]
    : [{ l: "Params", v: mem.params, c: T.params }, { l: "KV Cache", v: mem.kv, c: T.kv }]
  , [mode, mem]);

  const numNodes = Math.ceil(nGPU / 8);
  const nodeCols = numNodes <= 1 ? 1 : numNodes <= 2 ? 2 : numNodes <= 4 ? 2 : 4;
  const simIter = Math.floor(simProg / 100);
  const simTick = simProg % 100;

  // HFU estimation
  const hfuData = useMemo(() => estimateHFU(model, par, nGPU, batchSize, mp, ckpt), [model, par, nGPU, batchSize, mp, ckpt]);
  const iterTime = useMemo(() => estimateIterTime(model, par, nGPU, batchSize, hfuData.hfu, gpu.peak, ckpt), [model, par, nGPU, batchSize, hfuData.hfu, gpu.peak, ckpt]);
  const achievedTFLOPS = useMemo(() => hfuData.hfu * gpu.peak, [hfuData.hfu, gpu.peak]);
  // HFU for all paradigms (for comparison)
  const hfuAll = useMemo(() => PARADIGMS.map(p => ({
    id: p.id, name: p.name, c: p.c,
    ...estimateHFU(model, p.id, nGPU, batchSize, mp, ckpt),
  })), [model, nGPU, batchSize, mp, ckpt]);

  useEffect(() => {
    if (!simOn) { if (simRef.current) clearInterval(simRef.current); simRef.current = null; return; }
    simRef.current = setInterval(() => {
      setSimProg(p => { if (p >= TOTAL_SIM - 1) { setSimOn(false); return TOTAL_SIM - 1; } return p + 2; });
    }, 70);
    return () => { if (simRef.current) clearInterval(simRef.current); };
  }, [simOn]);

  const sel = { padding: "5px 8px", borderRadius: 6, border: `1px solid ${T.border}`, background: T.surface, color: T.text, fontSize: 12, ...mono, cursor: "pointer" };

  const pill = (active, c) => ({
    padding: "5px 13px", borderRadius: 18, fontSize: 12, fontWeight: 600, cursor: "pointer",
    background: active ? c + "14" : "transparent", color: active ? c : T.muted,
    border: `1.5px solid ${active ? c : T.border}`, transition: "all .15s", ...mono,
  });

  return (
    <div style={{ background: T.bg, minHeight: "100vh", fontFamily: "-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif", color: T.text }}>
      {/* HEADER */}
      <div style={{ background: T.surface, borderBottom: `1px solid ${T.border}`, padding: "8px 16px", display: "flex", flexWrap: "wrap", alignItems: "center", justifyContent: "space-between", gap: 8, position: "sticky", top: 0, zIndex: 40, boxShadow: T.shadow }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
          <span style={{ fontSize: 14, fontWeight: 700, ...mono }}><span style={{ color: pInfo.c }}>◆</span> Parallelism Explorer</span>
          <div style={{ display: "flex", borderRadius: 7, overflow: "hidden", border: `1px solid ${T.border}` }}>
            {["training", "inference"].map(m => (
              <button key={m} onClick={() => { setMode(m); setSimOn(false); setSimProg(0); }} style={{
                padding: "4px 14px", border: "none", cursor: "pointer",
                background: mode === m ? pInfo.c + "12" : T.surface,
                color: mode === m ? pInfo.c : T.muted, fontSize: 11, fontWeight: 600, ...mono,
              }}>{m === "training" ? "Training" : "Inference"}</button>
            ))}
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
          <select value={mId} onChange={e => setMId(e.target.value)} style={sel}>
            {MODELS.map(m => <option key={m.id} value={m.id}>{m.name} ({m.p}B)</option>)}
          </select>
          <select value={nGPU} onChange={e => setNGPU(+e.target.value)} style={sel}>
            {GPU_COUNTS.map(n => <option key={n} value={n}>{n} GPU{n > 1 ? "s" : ""}</option>)}
          </select>
          <select value={gId} onChange={e => setGId(e.target.value)} style={sel}>
            {GPUS.map(g => <option key={g.id} value={g.id}>{g.name}</option>)}
          </select>
          {mode === "training" && (
            <React.Fragment>
              <label style={{ display: "flex", alignItems: "center", gap: 3, fontSize: 10, color: T.muted, cursor: "pointer", ...mono }}>
                <input type="checkbox" checked={mp} onChange={e => setMp(e.target.checked)} /> FP16
              </label>
              <label style={{ display: "flex", alignItems: "center", gap: 3, fontSize: 10, color: T.muted, cursor: "pointer", ...mono }}>
                <input type="checkbox" checked={ckpt} onChange={e => setCkpt(e.target.checked)} /> ActCkpt
              </label>
            </React.Fragment>
          )}
          <span style={{ fontSize: 10, fontWeight: 700, padding: "3px 9px", borderRadius: 5, background: fits ? "#E2F4EA" : "#FCE8E8", color: fits ? T.ok : T.danger, ...mono }}>
            {fits ? "✓ Fits" : "✗ OOM"}
          </span>
        </div>
      </div>

      {/* PARADIGM TABS */}
      <div style={{ padding: "12px 16px 0", display: "flex", gap: 6, flexWrap: "wrap" }}>
        {PARADIGMS.map(p => (
          <div key={p.id} onClick={() => { setPar(p.id); setSimOn(false); setSimProg(0); }} style={pill(par === p.id, p.c)}>{p.name}</div>
        ))}
      </div>

      {/* CONCEPT + DEMO */}
      <div style={{ padding: "12px 16px", display: "grid", gridTemplateColumns: "minmax(0,1fr) minmax(0,1.3fr)", gap: 14, alignItems: "start" }}>
        {/* LEFT */}
        <div style={{ background: T.surface, borderRadius: 12, border: `1px solid ${T.border}`, padding: 18, boxShadow: T.shadow }}>
          <div style={{ fontSize: 9, fontWeight: 600, letterSpacing: ".07em", textTransform: "uppercase", color: pInfo.c, ...mono, marginBottom: 3 }}>How it works</div>
          <h2 style={{ fontSize: 17, fontWeight: 700, marginBottom: 8, lineHeight: 1.3 }}>{concept.title}</h2>
          {concept.pts.map((pt, i) => (
            <div key={i} style={{ display: "flex", gap: 7, marginBottom: 6, fontSize: 12.5, color: T.mid, lineHeight: 1.55 }}>
              <span style={{ color: pInfo.c, fontWeight: 700, flexShrink: 0, ...mono, fontSize: 11 }}>{i + 1}.</span>
              <span>{pt}</span>
            </div>
          ))}
          <div style={{ fontSize: 10, color: T.muted, ...mono, padding: "5px 8px", background: T.bg, borderRadius: 5, border: `1px solid ${T.borderSoft}`, marginTop: 6 }}>
            <strong style={{ color: pInfo.c }}>Comm:</strong> {concept.comm}
          </div>
          <div style={{ marginTop: 12, padding: "10px 4px", background: T.bg, borderRadius: 8, border: `1px solid ${T.borderSoft}`, display: "flex", justifyContent: "center", overflow: "auto" }}>
            <Diagram />
          </div>
        </div>

        {/* RIGHT */}
        <div>
          {par === "3d" && mem.tp && (
            <div style={{ background: T.surface, border: `1px solid ${T.border}`, borderRadius: 8, padding: "6px 12px", marginBottom: 8, display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap", fontSize: 12, ...mono, boxShadow: T.shadow }}>
              <span style={{ color: T.muted }}>Decomposition:</span>
              {[{ l: "TP", v: mem.tp, c: T.tp }, { l: "PP", v: mem.pp, c: T.pp }, { l: "DP", v: mem.dp, c: T.ddp }].map(d => (
                <span key={d.l} style={{ color: d.c, fontWeight: 700 }}>{d.l}={d.v}</span>
              ))}
              <span style={{ color: T.muted }}>= {nGPU}</span>
            </div>
          )}

          {/* GPU Grid */}
          <div style={{ display: "grid", gridTemplateColumns: `repeat(${nodeCols}, 1fr)`, gap: 8 }}>
            {Array.from({ length: numNodes }, (_, ni) => {
              const start = ni * 8;
              const end = Math.min(start + 8, nGPU);
              return (
                <div key={ni} style={{ background: T.surface, border: `1px solid ${T.border}`, borderRadius: 10, padding: 7, boxShadow: T.shadow }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                    <span style={{ fontSize: 9, fontWeight: 600, color: T.muted, ...mono }}>Node {ni}</span>
                    <span style={{ fontSize: 7, color: "#38A8C8", ...mono }}>● NVLink</span>
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: `repeat(${Math.min(4, end - start)}, 1fr)`, gap: 3 }}>
                    {Array.from({ length: end - start }, (_, j) => {
                      const gi = start + j;
                      const lbl = getLabel(par, gi, nGPU, model);
                      const sim = (simOn || simProg > 0) ? getSimGPU(gi, simTick, par, nGPU, perGPU, simIter) : null;
                      const dispMem = sim ? sim.mem : perGPU;
                      const dispSegs = sim
                        ? segs.map((s, si) => ({ ...s, v: s.v * (sim.mem / Math.max(perGPU, 0.01)) * (1 + (si === 3 ? 0.3 : si === 1 ? 0.2 : 0) * Math.sin(simTick * 0.06 + gi)) }))
                        : segs;
                      return (
                        <div key={gi} style={{ background: T.card, border: `1.5px solid ${fits ? T.border : T.danger + "60"}`, borderRadius: 7, padding: "5px 6px", borderTop: `2.5px solid ${pInfo.c}`, boxShadow: "0 1px 2px rgba(0,0,0,0.03)" }}>
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 2 }}>
                            <span style={{ fontSize: 9, fontWeight: 700, color: pInfo.c, ...mono }}>{lbl.top}</span>
                            <span style={{ fontSize: 8, color: T.muted, ...mono }}>{dispMem.toFixed(1)}G</span>
                          </div>
                          <MemBar segs={dispSegs} total={dispMem} cap={gpu.mem} h={11} />
                          {lbl.bot && <div style={{ fontSize: 7, color: T.muted, ...mono, textAlign: "center", marginTop: 1 }}>{lbl.bot}</div>}
                          {sim && (
                            <div style={{ marginTop: 3, borderTop: `1px solid ${T.borderSoft}`, paddingTop: 3 }}>
                              <TempBar temp={sim.temp} />
                              <div style={{ fontSize: 7, color: T.subtle, ...mono, textAlign: "center", marginTop: 1 }}>{sim.phase}</div>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
          {numNodes > 1 && (
            <div style={{ display: "flex", alignItems: "center", gap: 6, margin: "5px 0", justifyContent: "center" }}>
              <div style={{ flex: 1, maxWidth: 100, height: 1, background: T.threeD + "30" }} />
              <span style={{ fontSize: 7, color: T.threeD, ...mono }}>◆ InfiniBand</span>
              <div style={{ flex: 1, maxWidth: 100, height: 1, background: T.threeD + "30" }} />
            </div>
          )}
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginTop: 6, justifyContent: "center" }}>
            {segs.map((s, i) => (
              <div key={i} style={{ display: "flex", alignItems: "center", gap: 3 }}>
                <div style={{ width: 8, height: 8, borderRadius: 2, background: s.c, opacity: 0.8 }} />
                <span style={{ fontSize: 9, color: T.muted, ...mono }}>{s.l} {s.v.toFixed(1)}G</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* SIMULATION */}
      {mode === "training" && (
        <div style={{ padding: "0 16px 14px" }}>
          <div style={{ background: T.surface, border: `1px solid ${T.border}`, borderRadius: 12, padding: 16, boxShadow: T.shadow }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8, marginBottom: 12 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
                <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: ".07em", textTransform: "uppercase", color: T.muted, ...mono }}>Training Simulation</span>
                <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
                  <span style={{ fontSize: 11, color: T.mid, ...mono }}>Batch:</span>
                  {[1, 2, 4, 8, 16, 32].map(b => (
                    <button key={b} onClick={() => { setBatchSize(b); setSimOn(false); setSimProg(0); }}
                      style={{ padding: "2px 8px", borderRadius: 4, border: `1px solid ${batchSize === b ? pInfo.c : T.border}`, background: batchSize === b ? pInfo.c + "12" : "transparent", color: batchSize === b ? pInfo.c : T.muted, fontSize: 10, ...mono, cursor: "pointer", fontWeight: 600 }}>
                      {b}
                    </button>
                  ))}
                </div>
              </div>
              <div style={{ display: "flex", gap: 6 }}>
                {!simOn && simProg < TOTAL_SIM - 1 && (
                  <button onClick={() => { setSimProg(0); setSimOn(true); }} style={{ padding: "5px 16px", borderRadius: 6, border: "none", background: pInfo.c, color: "#fff", fontSize: 11, fontWeight: 600, ...mono, cursor: "pointer" }}>▶ Run 10 Iterations</button>
                )}
                {simOn && (
                  <button onClick={() => setSimOn(false)} style={{ padding: "5px 16px", borderRadius: 6, border: `1px solid ${T.border}`, background: T.surface, color: T.mid, fontSize: 11, fontWeight: 600, ...mono, cursor: "pointer" }}>⏸ Pause</button>
                )}
                {simProg > 0 && !simOn && (
                  <button onClick={() => setSimProg(0)} style={{ padding: "5px 12px", borderRadius: 6, border: `1px solid ${T.border}`, background: T.surface, color: T.muted, fontSize: 11, ...mono, cursor: "pointer" }}>↺ Reset</button>
                )}
              </div>
            </div>

            {/* Progress */}
            <div style={{ marginBottom: 10 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                <span style={{ fontSize: 10, color: T.mid, ...mono }}>Iteration {simIter + 1}/10</span>
                <span style={{ fontSize: 10, color: T.muted, ...mono }}>
                  Phase: {simTick < 25 ? "Forward" : simTick < 55 ? "Backward" : simTick < 78 ? "Communication" : "Optimizer Step"}
                </span>
              </div>
              <div style={{ height: 6, borderRadius: 3, background: T.free, overflow: "hidden" }}>
                <div style={{ width: `${(simProg / TOTAL_SIM) * 100}%`, height: "100%", borderRadius: 3, background: pInfo.c, transition: "width .15s", opacity: 0.7 }} />
              </div>
              <div style={{ display: "flex", gap: 2, marginTop: 4 }}>
                {Array.from({ length: 10 }, (_, i) => (
                  <div key={i} style={{ flex: 1, height: 3, borderRadius: 1.5, background: i < simIter ? pInfo.c : i === simIter ? (pInfo.c + "60") : T.free, transition: "background .3s" }} />
                ))}
              </div>
            </div>

            {/* Phase bar */}
            <div style={{ display: "flex", gap: 2, marginBottom: 10 }}>
              {[{ name: "Forward", pct: 25, c: T.params }, { name: "Backward", pct: 30, c: T.grads }, { name: "Comm", pct: 23, c: T.fsdp }, { name: "Optim", pct: 22, c: T.optim }].map((ph, i) => {
                const isActive = (i === 0 && simTick < 25) || (i === 1 && simTick >= 25 && simTick < 55) || (i === 2 && simTick >= 55 && simTick < 78) || (i === 3 && simTick >= 78);
                return (
                  <div key={i} style={{ flex: ph.pct, textAlign: "center", padding: "4px 0", borderRadius: 4, background: isActive ? ph.c + "18" : T.free, border: `1px solid ${isActive ? ph.c + "40" : T.borderSoft}`, transition: "all .2s" }}>
                    <span style={{ fontSize: 9, ...mono, fontWeight: isActive ? 700 : 500, color: isActive ? ph.c : T.muted }}>{ph.name}</span>
                  </div>
                );
              })}
            </div>

            {/* GPU temp strip */}
            {(simOn || simProg > 0) ? (
              <div>
                <div style={{ fontSize: 9, fontWeight: 600, letterSpacing: ".06em", textTransform: "uppercase", color: T.muted, ...mono, marginBottom: 6 }}>GPU Load & Temperature</div>
                <div style={{ display: "grid", gridTemplateColumns: `repeat(${Math.min(nGPU, 8)}, 1fr)`, gap: 4 }}>
                  {Array.from({ length: Math.min(nGPU, 16) }, (_, gi) => {
                    const sim = getSimGPU(gi, simTick, par, nGPU, perGPU, simIter);
                    return (
                      <div key={gi} style={{ background: T.bg, borderRadius: 6, padding: "5px 6px", border: `1px solid ${T.borderSoft}` }}>
                        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
                          <span style={{ fontSize: 8, fontWeight: 600, color: pInfo.c, ...mono }}>GPU {gi}</span>
                          <span style={{ fontSize: 8, color: T.muted, ...mono }}>{sim.mem.toFixed(1)}G</span>
                        </div>
                        <div style={{ height: 5, borderRadius: 3, background: T.free, overflow: "hidden", marginBottom: 3 }}>
                          <div style={{ width: `${Math.min(100, (sim.mem / gpu.mem) * 100)}%`, height: "100%", borderRadius: 3, background: sim.mem > gpu.mem ? T.danger : pInfo.c, transition: "width .2s", opacity: 0.7 }} />
                        </div>
                        <TempBar temp={sim.temp} />
                        <div style={{ fontSize: 7, color: T.subtle, ...mono, textAlign: "center", marginTop: 2 }}>
                          {sim.phase === "idle" ? "idle" : sim.phase === "forward" ? "fwd" : sim.phase === "backward" ? "bwd" : sim.phase === "comm" ? "comm" : "optim"}
                        </div>
                      </div>
                    );
                  })}
                </div>
                {nGPU > 16 && <div style={{ fontSize: 9, color: T.muted, ...mono, marginTop: 4, textAlign: "center" }}>Showing 16 of {nGPU} GPUs</div>}
                <div style={{ display: "flex", gap: 8, marginTop: 8, justifyContent: "center", flexWrap: "wrap" }}>
                  {[{ t: 40, l: "Cool" }, { t: 55, l: "Warm" }, { t: 70, l: "Active" }, { t: 80, l: "Hot" }, { t: 90, l: "Peak" }].map(x => (
                    <div key={x.t} style={{ display: "flex", alignItems: "center", gap: 3 }}>
                      <div style={{ width: 8, height: 8, borderRadius: 2, background: tempColor(x.t) }} />
                      <span style={{ fontSize: 8, color: T.muted, ...mono }}>{x.l} ({x.t}°C)</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div style={{ textAlign: "center", padding: "14px 0", color: T.subtle, fontSize: 12 }}>
                Press <strong style={{ color: pInfo.c }}>Run</strong> to simulate 10 training iterations — watch memory fluctuate and GPU temperatures respond to each phase.
              </div>
            )}
          </div>
        </div>
      )}

      {/* HFU ANALYSIS */}
      {mode === "training" && (
        <div style={{ padding: "0 16px 14px" }}>
          <div style={{ background: T.surface, border: `1px solid ${T.border}`, borderRadius: 12, padding: 16, boxShadow: T.shadow }}>
            <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: ".07em", textTransform: "uppercase", color: T.muted, ...mono, marginBottom: 12 }}>
              Hardware FLOP Utilization (HFU)
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "minmax(0,1fr) minmax(0,1fr)", gap: 16 }}>
              {/* LEFT — Breakdown waterfall */}
              <div>
                <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginBottom: 10 }}>
                  <span style={{ fontSize: 28, fontWeight: 800, color: hfuData.hfu > 0.4 ? T.ok : hfuData.hfu > 0.25 ? T.warn : T.danger, ...mono }}>
                    {(hfuData.hfu * 100).toFixed(1)}%
                  </span>
                  <span style={{ fontSize: 11, color: T.muted }}>estimated HFU</span>
                </div>
                {/* Big HFU bar */}
                <div style={{ height: 10, borderRadius: 5, background: T.free, overflow: "hidden", marginBottom: 12 }}>
                  <div style={{ width: `${hfuData.hfu * 100}%`, height: "100%", borderRadius: 5, background: hfuData.hfu > 0.4 ? T.ok : hfuData.hfu > 0.25 ? T.warn : T.danger, transition: "width .4s" }} />
                </div>
                {/* Waterfall breakdown */}
                <div style={{ fontSize: 10, color: T.muted, ...mono, marginBottom: 6 }}>Efficiency breakdown (multiplicative):</div>
                {[
                  { label: "Compute (matmul occupancy)", val: hfuData.compute, tip: `H=${model.H} → ${(hfuData.compute * 100).toFixed(0)}%` },
                  { label: "Communication overhead", val: hfuData.comm, tip: `${par.toUpperCase()} on ${nGPU} GPUs` },
                  { label: "Pipeline efficiency", val: hfuData.pipe, tip: par === "pp" || par === "3d" ? "bubble ratio" : "no pipeline" },
                  { label: "Batch utilization", val: hfuData.batch, tip: `batch=${batchSize}` },
                  { label: "Precision factor", val: hfuData.mpFactor, tip: mp ? "FP16 tensor cores" : "FP32 — 2× slower" },
                ].map((row, i) => (
                  <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                    <div style={{ width: 140, fontSize: 10, color: T.mid, ...mono, flexShrink: 0 }}>{row.label}</div>
                    <div style={{ flex: 1, height: 8, borderRadius: 4, background: T.free, overflow: "hidden" }}>
                      <div style={{ width: `${row.val * 100}%`, height: "100%", borderRadius: 4, background: row.val > 0.8 ? T.ok : row.val > 0.55 ? T.warn : T.danger, transition: "width .3s", opacity: 0.7 }} />
                    </div>
                    <span style={{ fontSize: 10, fontWeight: 700, color: T.text, ...mono, minWidth: 34, textAlign: "right" }}>{(row.val * 100).toFixed(0)}%</span>
                    <span style={{ fontSize: 8, color: T.subtle, ...mono, minWidth: 60 }}>{row.tip}</span>
                  </div>
                ))}
                {/* Achieved TFLOPS and time */}
                <div style={{ marginTop: 10, padding: "8px 10px", background: T.bg, borderRadius: 6, border: `1px solid ${T.borderSoft}` }}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, ...mono, marginBottom: 2 }}>
                    <span style={{ color: T.muted }}>Achieved per GPU</span>
                    <span style={{ color: T.text, fontWeight: 700 }}>{achievedTFLOPS.toFixed(0)} / {gpu.peak} TFLOPS</span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, ...mono, marginBottom: 2 }}>
                    <span style={{ color: T.muted }}>Model FLOPs/iter</span>
                    <span style={{ color: T.text, fontWeight: 700 }}>{(() => {
                      let gb = batchSize * nGPU;
                      if (par === "tp") gb = batchSize;
                      else if (par === "3d") { const tp2=Math.min(8,nGPU), pp2=Math.max(1,Math.min(nGPU/tp2,Math.ceil(model.L/8))); gb = batchSize * Math.max(1, nGPU/(tp2*pp2)); }
                      return ((ckpt ? 8 : 6) * model.p * gb * model.S / 1e3).toFixed(1);
                    })()} PetaFLOPs</span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, ...mono }}>
                    <span style={{ color: T.muted }}>Est. time/iteration</span>
                    <span style={{ color: pInfo.c, fontWeight: 700 }}>{iterTime < 1 ? `${(iterTime * 1000).toFixed(0)} ms` : `${iterTime.toFixed(2)} s`}</span>
                  </div>
                </div>
              </div>

              {/* RIGHT — Paradigm comparison */}
              <div>
                <div style={{ fontSize: 10, color: T.muted, ...mono, marginBottom: 8 }}>Compare paradigms (same model + cluster):</div>
                <svg viewBox="0 0 320 200" width="100%" style={{ maxWidth: 360 }}>
                  {hfuAll.map((h, i) => {
                    const barY = 8 + i * 38;
                    const barW = Math.max(4, h.hfu * 400);
                    const isActive = h.id === par;
                    return (
                      <g key={h.id}>
                        <text x={0} y={barY + 13} fill={isActive ? h.c : T.muted} style={{ fontSize: 11, ...mono, fontWeight: isActive ? 700 : 500 }}>{h.name}</text>
                        <rect x={52} y={barY} width={240} height={22} rx={4} fill={T.free} />
                        <rect x={52} y={barY} width={Math.min(240, barW)} height={22} rx={4} fill={h.c} opacity={isActive ? 0.6 : 0.25} />
                        {isActive && <rect x={52} y={barY} width={Math.min(240, barW)} height={22} rx={4} fill="none" stroke={h.c} strokeWidth={1.5} />}
                        <text x={56 + Math.min(240, barW)} y={barY + 14} fill={isActive ? T.text : T.muted} style={{ fontSize: 11, ...mono, fontWeight: 700 }}>{(h.hfu * 100).toFixed(1)}%</text>
                      </g>
                    );
                  })}
                  <text x={52} y={200} fill={T.subtle} style={{ fontSize: 8, ...mono }}>0%</text>
                  <text x={280} y={200} fill={T.subtle} style={{ fontSize: 8, ...mono }}>60%</text>
                  <line x1={52} y1={192} x2={292} y2={192} stroke={T.border} strokeWidth={0.5} />
                  {[0, 15, 30, 45, 60].map(v => (
                    <line key={v} x1={52 + v * 4} y1={0} x2={52 + v * 4} y2={190} stroke={T.borderSoft} strokeWidth={0.5} />
                  ))}
                </svg>
                <div style={{ fontSize: 10, color: T.subtle, ...mono, lineHeight: 1.6, marginTop: 4 }}>
                  <div>HFU = Achieved TFLOPS / Peak {gpu.peak} TFLOPS</div>
                  <div>FLOPs/iter = {ckpt ? "8" : "6"} × P × global_batch × seq_len</div>
                  <div style={{ marginTop: 4, color: T.muted, fontSize: 9 }}>
                    Typical real-world HFU: 30–55% depending on model size, cluster, and interconnect topology.
                    Larger models and batch sizes generally achieve higher utilization.
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* STATS */}
      <div style={{ padding: "0 16px 14px", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 8 }}>
        {[
          { label: "Per-GPU Memory", value: `${perGPU.toFixed(1)} / ${gpu.mem} GB`, color: fits ? T.text : T.danger, sub: `${(perGPU / gpu.mem * 100).toFixed(0)}% capacity` },
          { label: "Model (params)", value: `${(model.p * (mode === "training" ? (mp ? 2 : 4) : 2)).toFixed(1)} GB`, color: T.params, sub: `${model.p}B × ${mp ? "2B" : "4B"}` },
          ...(mode === "training" ? [
            { label: "Optimizer / GPU", value: `${mem.optim.toFixed(1)} GB`, color: T.optim, sub: "Adam: master + m + v" },
            { label: "Memory Savings", value: par === "ddp" ? "0%" : `${Math.max(0, (100 - perGPU / (mem.totP + mem.totG + mem.totO + mem.totA) * 100)).toFixed(0)}%`, color: T.ok, sub: par === "ddp" ? "DDP replicates all" : "vs single-GPU" },
            { label: "HFU", value: `${(hfuData.hfu * 100).toFixed(1)}%`, color: hfuData.hfu > 0.4 ? T.ok : hfuData.hfu > 0.25 ? T.warn : T.danger, sub: `${achievedTFLOPS.toFixed(0)} / ${gpu.peak} TFLOPS` },
          ] : [
            { label: "KV Cache / GPU", value: `${mem.kv.toFixed(1)} GB`, color: T.kv, sub: `seq=${model.S}` },
          ]),
          { label: "Cluster", value: `${nGPU}× ${gpu.name.split(" ")[0]}`, color: T.text, sub: `${numNodes} node${numNodes > 1 ? "s" : ""}` },
        ].map((s, i) => (
          <div key={i} style={{ background: T.surface, border: `1px solid ${T.border}`, borderRadius: 9, padding: "10px 12px", boxShadow: T.shadow }}>
            <div style={{ fontSize: 8, fontWeight: 600, letterSpacing: ".06em", textTransform: "uppercase", color: T.muted, ...mono }}>{s.label}</div>
            <div style={{ fontSize: 17, fontWeight: 700, color: s.color, ...mono, lineHeight: 1.3, margin: "2px 0" }}>{s.value}</div>
            <div style={{ fontSize: 10, color: T.subtle }}>{s.sub}</div>
          </div>
        ))}
      </div>

      {/* MEMORY MATH */}
      <div style={{ padding: "0 16px 20px" }}>
        <div style={{ background: T.surface, border: `1px solid ${T.border}`, borderRadius: 10, padding: 14, boxShadow: T.shadow }}>
          <div style={{ fontSize: 8, fontWeight: 600, letterSpacing: ".06em", textTransform: "uppercase", color: T.muted, ...mono, marginBottom: 6 }}>
            Memory math — {mode} · {pInfo.name} · {nGPU} GPUs · batch={batchSize}
          </div>
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
            <div style={{ background: T.bg, borderRadius: 7, padding: "8px 12px", flex: "1 1 180px", display: "flex", flexDirection: "column", gap: 3 }}>
              {segs.map((s, i) => (
                <div key={i} style={{ display: "flex", justifyContent: "space-between", fontSize: 11, ...mono }}>
                  <span style={{ color: s.c, fontWeight: 600 }}>{s.l}</span>
                  <span style={{ color: T.text }}>{s.v.toFixed(2)} GB</span>
                </div>
              ))}
              <div style={{ borderTop: `1px solid ${T.border}`, paddingTop: 3, display: "flex", justifyContent: "space-between", fontSize: 12, fontWeight: 700, ...mono }}>
                <span>Total</span>
                <span style={{ color: fits ? T.ok : T.danger }}>{perGPU.toFixed(2)} GB</span>
              </div>
            </div>
            <div style={{ fontSize: 11.5, color: T.mid, lineHeight: 1.7, flex: "1 1 260px" }}>
              {mode === "training" ? (
                <React.Fragment>
                  <div><strong style={{ ...mono, color: T.params }}>Params</strong> = {model.p}B × {mp ? "2" : "4"}B{par !== "ddp" ? ` ÷ ${nGPU}` : ""} = <strong style={mono}>{mem.params.toFixed(1)} GB</strong></div>
                  <div><strong style={{ ...mono, color: T.grads }}>Grads</strong> = {model.p}B × {mp ? "2" : "4"}B{par !== "ddp" ? ` ÷ ${nGPU}` : ""} = <strong style={mono}>{mem.grads.toFixed(1)} GB</strong></div>
                  <div><strong style={{ ...mono, color: T.optim }}>Optim</strong> = {model.p}B × {mp ? "12" : "8"}B{par !== "ddp" ? ` ÷ ${nGPU}` : ""} = <strong style={mono}>{mem.optim.toFixed(1)} GB</strong></div>
                  <div><strong style={{ ...mono, color: T.act }}>Act</strong> ≈ {batchSize}×{model.S}×{model.H}×{model.L} = <strong style={mono}>{mem.act.toFixed(1)} GB</strong></div>
                </React.Fragment>
              ) : (
                <React.Fragment>
                  <div><strong style={{ ...mono, color: T.params }}>Params</strong> = {model.p}B × 2B{par !== "ddp" ? ` ÷ ${nGPU}` : ""} = <strong style={mono}>{mem.params.toFixed(1)} GB</strong></div>
                  <div><strong style={{ ...mono, color: T.kv }}>KV</strong> = 4×{model.L}L×{model.H}H×{model.S}S×4b×2B = <strong style={mono}>{mem.kv.toFixed(1)} GB</strong></div>
                </React.Fragment>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
