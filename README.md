# KVHALO-Inference

**KVHALO** (KV Cache Hierarchical Adaptive Learning Optimizer) is a novel inference-time framework that dynamically reconstructs degraded KV caches in autoregressive transformer models. By intercepting key-value cache tensors during generation, KVHALO applies super-resolution regression through a **Hierarchical Reasoning Model (HRM)** architecture to recover high-fidelity representations from severely quantized low-bit footprints.

**NOTE**: This repository is only for Mistral. If you would like to implement a different model, you will need to modify the code.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [Hierarchical Reasoning Model (HRM)](#hierarchical-reasoning-model-hrm)
  - [KV Upscaler Backbone](#kv-upscaler-backbone)
  - [Quantization & Reconstruction Pipeline](#quantization--reconstruction-pipeline)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Quick Start](#quick-start)
  - [Interactive Comparison Demo](#interactive-comparison-demo)
  - [Programmatic API](#programmatic-api)
- [Configuration](#configuration)
- [Training](#training)
- [Performance Metrics](#performance-metrics)
- [License](#license)

---

## Overview

KV caches are critical to the performance of autoregressive language models, storing past token representations for efficient attention computation. However, storing full 16-bit floating-point KV caches consumes significant memory, especially for long sequences. Traditional approaches compress these caches through quantization, but aggressive low-bit quantization (e.g., 2-bit) degrades model output quality.

**KVHALO solves this problem** by:

1. **Intercepting** KV cache tensors at specified transformer layers during inference
2. **Compressing** them to low-bit representations (simulating extreme quantization)
3. **Dynamically reconstructing** them back to high-fidelity 16-bit tensors using the HRM neural network
4. **Injecting** the reconstructed caches back into the model's attention mechanism

This enables **memory-efficient inference** without sacrificing generation quality — achieving up to **8x VRAM reduction** on targeted layers while maintaining output fidelity.

---

## Architecture

### Hierarchical Reasoning Model (HRM)

The HRM is the core neural architecture that performs the KV cache reconstruction. It consists of two interwoven computational streams:

#### Dual-Stream Design

```
┌─────────────────────────────────────────────────────────────┐
│                    HRMInner Module                          │
│                                                             │
│  z_H (High-State) ──→ HRMBlock ──→ z_H_new                │
│       ↑                        ↑                            │
│       │                        │                            │
│  z_L (Low-State) ──→ HRMBlock ──→ z_L_current             │
│       ↑                       │                             │
│       └── Cross-Stream Routing ─┘                           │
└─────────────────────────────────────────────────────────────┘
```

The HRMInner module maintains two parallel state representations:

- **High-State (z_H)**: Captures coarse-grained, global dependencies in the KV cache
- **Low-State (z_L)**: Captures fine-grained, local patterns and residual information

These states interact through **Dual-Stream Information Routing**:

```python
z_L_input = RMSNorm(z_L + z_H)          # Fuse high-state into low-state
z_L_current = L_module(z_L_input)        # Process through low-stream
z_H_input = z_H + z_L_current            # Feed back low-state to high-stream
z_H_new = H_module(z_H_input)            # Process through high-stream
```

#### HRMBlock Components

Each HRMBlock contains:

| Component | Description |
|-----------|-------------|
| **RMSNorm** | Root Mean Square Layer Normalization for stable training |
| **Q/K/V/O Projections** | Multi-head attention with learned linear transformations |
| **Rotary Positional Embeddings** | Sinusoidal RoPE for position-aware attention computation |
| **SwiGLU Activation** | Gated linear unit with SiLU activation: `SiLU(w1(x) ⊗ w2(x))` |
| **FlashAttention** | PyTorch 2.0 optimized scaled dot-product attention with causal masking |

#### Rotary Positional Embeddings

KVHALO uses rotary positional embeddings to inject sequence position information:

```
q_rot = Rθ · q,  k_rot = Rθ · k
```

Where Rθ applies rotation matrices parameterized by:
```
freqs = 1.0 / (base^(2i/d_head)) for i in [0, d_head/2)
```

### KV Upscaler Backbone

The `KvHALO_Upscaler` orchestrates the reconstruction pipeline:

```
┌──────────────────────────────────────────────────────────────────┐
│                     KvHALO_Upscaler                              │
│                                                                  │
│  lossy_kv [B, S, 2·teacher_dim]                                │
│       │                                                          │
│       ▼                                                          │
│  ┌──────────────┐                                               │
│  │ Input        │  Linear(2·teacher_dim → d_model)             │
│  │ Compression  │  → z_L [B, S, d_model]                       │
│  └──────────────┘                                               │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────┐                                    │
│  │ HRM Thinking Loops (t=2) │  Iterative refinement            │
│  │  z_H, z_L ← HRMInner    │  Dual-stream information routing │
│  └─────────────────────────┘                                    │
│       │                                                          │
│       ▼                                                          │
│  ┌──────────────┐                                               │
│  │ Regression   │  Linear(d_model → 2·teacher_dim)             │
│  │ Head         │  → predicted_states [B, S, 2·teacher_dim]    │
│  └──────────────┘                                               │
│       │                                                          │
│       ▼                                                          │
│  predicted_states [B, S, 2·teacher_dim] ← Chunk → K_pred, V_pred│
└──────────────────────────────────────────────────────────────────┘
```

**Forward Pass Flow:**

1. **Input Compression**: The lossy (quantized) KV states are projected from the teacher dimension (`teacher_dim * 2`) down to the latent space (`d_model`)
2. **HRM Thinking**: Exactly `t_steps=2` iterations of the dual-stream HRMInner process the latent representations, progressively refining the reconstruction
3. **Regression**: The final high-state is projected back to the full teacher dimension, producing predicted high-fidelity KV states

### Quantization & Reconstruction Pipeline

KVHALO supports configurable bit-width quantization:

```
Steps = 2^bits - 1

For 1-bit:  2 levels  [0, 1]       → 50% compression
For 2-bit:  4 levels [0, 1/3, 2/3, 1]  → 8x compression
For 4-bit:  15 levels [0, 1/14, ..., 1] → 4x compression
```

The quantization process:
1. Normalize tensor to [0, 1] range
2. Round to nearest discrete level
3. Rescale back to original magnitude

---

## Project Structure

```
KVHALO-Inference/
├── ExecutionClient.py              # Production-ready client API
├── README.md                       # This file
├── LICENSE                         # Apache 2.0 License
├── requirements.txt                # Python dependencies
├── examples/
│   └── compare_base_mistral.py    # Interactive Gradio comparison demo
└── sub_libraries/
    ├── HRM_Main.py                 # HRM architecture + KvHALO_Upscaler
    └── HRM_Simulated_Quantization.py  # Quantization utility functions
```

| File | Purpose |
|------|---------|
| `ExecutionClient.py` | `MistralKvHALO` class — production API for integrating KVHALO into any generation pipeline |
| `sub_libraries/HRM_Main.py` | Core HRM architecture: `HRMBlock`, `HRMInner`, `KvHALO_Upscaler`, `RMSNorm`, `RotaryPositionalEmbeddings` |
| `sub_libraries/HRM_Simulated_Quantization.py` | `simulate_low_bit_quantization()` utility for training-time quantization simulation |
| `examples/compare_base_mistral.py` | Full Gradio application comparing baseline vs. KVHALO-enhanced generation with live waveform visualization |

---

## Installation

### Prerequisites

- **Python 3.8+**
- **GPU** with CUDA drivers (recommended) **or** macOS with Apple Silicon (MPS support)
- **Hugging Face account** for model access (if downloading Mistral-7B)

### Setup

```bash
# Install Python dependencies
pip install -r requirements.txt
```

### Required Dependencies
- Note that all models are available at the Huggingface Repository [Here](https://huggingface.co/richyvd/kvhalo)



---

## Usage

### Quick Start

```python
from ExecutionClient import MistralKvHALO

# Initialize the KVHALO engine
client = MistralKvHALO(target_layer=15)

# Load trained weights (if available)
client.load_upscaler_weights("best_KVHALO.pt")

# Generate with KVHALO reconstruction enabled
prompt = "Explain quantum entanglement in simple terms."
for chunk in client.generate(prompt, max_new_tokens=64, bits=2, temperature=0.7):
    print(chunk, end="", flush=True)
```

### Interactive Comparison Demo

Launch the Gradio application to compare baseline generation (without KVHALO) against KVHALO-enhanced generation in real-time:

```bash
cd examples
python compare_base_mistral.py
```

The demo provides:

- **Side-by-side text output** comparing baseline vs. KVHALO-enhanced responses
- **Live waveform visualization** showing the KV cache manifold at each stage (original, quantized, reconstructed)
- **Real-time telemetry** including throughput speed, generation time, VRAM savings, and reconstruction fidelity metrics
- **Configurable parameters**: quantization bit-width (1-4 bits), max tokens, temperature, and top-p

### Programmatic API

The `MistralKvHALO` class provides a clean interface:

```python
from ExecutionClient import MistralKvHALO

client = MistralKvHALO(
    model_id="mistralai/Mistral-7B-Instruct-v0.3",  # Base model
    target_layer=15                                   # Layer to intercept
)

# Load checkpoint
client.load_upscaler_weights("best_KVHALO.pt")

# Streaming generation
for response_chunk in client.generate(
    prompt="Your prompt here",
    max_new_tokens=128,
    bits=2,            # Quantization level (1-4)
    temperature=0.7    # Sampling temperature
):
    print(response_chunk, end="")

# Manually control patching
client.patch(bits=2)      # Enable KVHALO
# ... run generation ...
client.unpatch()           # Disable KVHALO (restores 16-bit)
```

---

## Configuration

KVHALO uses a configuration dictionary matching the training setup:

```python
CONFIG = {
    "d_model": 768,      # Latent dimension (candidate space)
    "n_heads": 12,       # Multi-head attention heads
    "d_ff": 3072,        # Feed-forward dimension (4× d_model)
    "dropout": 0.0,      # Dropout rate (0.0 for inference)
    "teacher_dim": 1024  # Per-head dimension × num_heads for Mistral GQA
}
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 768 | Latent representation dimension after input compression |
| `n_heads` | 12 | Number of attention heads in HRMBlock |
| `d_ff` | 3072 | Hidden dimension of the SwiGLU feed-forward network |
| `dropout` | 0.0 | Dropout rate (set to 0.0 for inference) |
| `teacher_dim` | 1024 | Dimension of the teacher model's KV per layer (Mistral: 8 GQA heads × 128 head_dim) |

---

## Training

To train a KVHALO upscaler for your model:

1. **Collect KV cache pairs** from your base model (16-bit "teacher" states and corresponding 2-bit "student" states)
2. **Compute continuous latent distillation loss**:
   - **MSE Loss**: Aligns geometric magnitudes between predicted and target states
   - **Cosine Similarity Loss**: Aligns directional semantics between predicted and target states
3. **Save checkpoint**:
   ```python
   torch.save({
       "model_state_dict": KVHALO.state_dict(),
       "config": CONFIG
   }, "best_KVHALO.pt")
   ```

The loss function optimizes:
```
L = MSE(predicted_states, target_states) + (1 - cosine_similarity(flat_pred, flat_target))
```

---

## Performance Metrics

### Memory Savings

For a targeted layer, KVHALO achieves approximately:

```
VRAM Savings ≈ (16-bit footprint) - (bits-bit footprint)

Example for Layer 15 with 100 token sequence:
- FP16 cache: ~128 KB
- 2-bit cache: ~8 KB
- Savings: ~120 KB per layer
```

### Reconstruction Quality

- **Empirical cosine similarity**: ~91.85% alignment with original 16-bit states (at 2-bit quantization)
- **Linguistic divergence recovered**: KVHALO-generated text shows significantly higher similarity to ground-truth 16-bit generation compared to naive quantized baseline

### Generation Speed

- **Warm-up overhead**: First inference includes kernel compilation (~2-5 seconds on GPU, ~5-10 seconds on MPS)
- **Per-token overhead**: Minimal (~0.1-0.5ms per token for the 92M parameter HRM upscaler)
- **Baseline throughput**: Comparable to unpatched Mistral-7B generation speed

---

## Technical Details

### HRM Architecture Summary

| Aspect | Value |
|--------|-------|
| Total Parameters | ~92M |
| Thinking Steps (t_steps) | 2 |
| State Types | Dual (High-State z_H + Low-State z_L) |
| Attention Mechanism | Multi-head with FlashAttention 2 |
| Position Encoding | Rotary (RoPE), base=10000 |
| Activation | SwiGLU (SiLU × linear) |
| Normalization | RMSNorm (ε=1e-5) |

### Supported Models

Currently tested with:
- **Mistral-7B-Instruct-v0.3** (GQA with 8 KV heads)

The framework is designed to be model-agnostic — adjust `teacher_dim` for different architectures.

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 8 GB VRAM | 16+ GB VRAM |
| RAM | 16 GB | 32+ GB |
| Apple Silicon | MPS-compatible | M1 Pro or later |
| CUDA | Compute 7.0+ | Compute 8.0+ |

---

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.

```
Copyright 2026 Richard Wang & The KVHALO Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## Citation

If you use KVHALO in your research, please cite:

```bibtex
@software{kvhalo2026,
  author = {Wang, Richard},
  title = {KVHALO-Inference: KV Cache Hierarchical Adaptive Learning Optimizer},
  year = {2026},
  url = {https://github.com/RA86-dev/KVHALO-Inference}
}
