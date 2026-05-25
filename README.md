
# KVHALO-Inference

**KVHALO** (KV Cache Hierarchical Adaptive Learning Optimizer) is a novel inference-time framework that dynamically reconstructs degraded KV caches in autoregressive transformer models. By intercepting key-value cache tensors during generation, KVHALO applies super-resolution regression through a **Hierarchical Reasoning Model (HRM)** architecture to recover high-fidelity representations from severely quantized, low-bit footprints.

> ⚠️ **NOTE**: This implementation is currently optimized for **Mistral** architectures. Adapting it to other models requires minor modifications to the layer interception hooks.
Training code is located at [RA86-dev/KVHALO_Training](https://github.com/RA86-dev/KVHALO_Training)
---


## Overview

KV caches are critical to the performance of autoregressive language models, storing past token representations for efficient attention computation. However, storing full 16-bit floating-point (FP16) KV caches consumes significant memory, especially during long-context generation. Traditional approaches compress these caches through quantization, but aggressive low-bit quantization (e.g., 2-bit) severely degrades model output quality.

**KVHALO solves this bottleneck** by executing a four-step pipeline:

1. **Intercepting:** Catches KV cache tensors at specified transformer layers during inference.
2. **Compressing:** Simulates extreme, low-bit quantization configurations.
3. **Reconstructing:** Dynamically upscales the low-bit footprints back to high-fidelity 16-bit tensors using the HRM architecture.
4. **Injecting:** Feeds the reconstructed high-fidelity caches back into the model's attention mechanism natively.

This enables **memory-efficient inference** without sacrificing generation quality—achieving up to an **8x VRAM reduction** on targeted layers while maintaining baseline output fidelity.

---

## Architecture

### Hierarchical Reasoning Model (HRM)

The HRM is the core neural architecture performing the KV cache reconstruction. It utilizes two interwoven computational streams to capture macro- and micro-level context features.

#### Dual-Stream Design

```
┌─────────────────────────────────────────────────────────────┐
│                   HRMInner Module                           │
│                                                             │
│  z_H (High-State) ──→ HRMBlock ──→ z_H_new                  │
│       ↑                        ↑                            │
│       │                        │                            │
│  z_L (Low-State)  ──→ HRMBlock ──→ z_L_current              │
│       ↑                        │                            │
│       └── Cross-Stream Routing ─┘                            │
└─────────────────────────────────────────────────────────────┘

```

The `HRMInner` module maintains two parallel state representations:

* **High-State ($z_H$):** Captures coarse-grained, global dependencies in the KV cache.
* **Low-State ($z_L$):** Captures fine-grained, local patterns and residual information.

These states interact dynamically through **Dual-Stream Information Routing**:

```python
z_L_input = RMSNorm(z_L + z_H)          # Fuse high-state into low-state
z_L_current = L_module(z_L_input)        # Process through low-stream
z_H_input = z_H + z_L_current            # Feed back low-state to high-stream
z_H_new = H_module(z_H_input)            # Process through high-stream

```

#### HRMBlock Components

Each `HRMBlock` utilizes highly optimized, modern architectural components:

| Component | Description |
| --- | --- |
| **RMSNorm** | Root Mean Square Layer Normalization for stable, gradient-bounded training. |
| **Q/K/V/O Projections** | Multi-head attention layers mapped with learned linear transformations. |
| **Rotary Embeddings** | Sinusoidal RoPE for position-aware attention computation. |
| **SwiGLU Activation** | Gated linear unit utilizing SiLU activation: $\text{SiLU}(W_1(x)) \times W_2(x)$. |
| **FlashAttention** | PyTorch 2.0 optimized scaled dot-product attention with causal masking. |

#### Rotary Positional Embeddings

To inject sequence position data, KVHALO maps rotary positional coordinates onto query and key states:

$$q_{\text{rot}} = R_{\Theta} \times q, \quad k_{\text{rot}} = R_{\Theta} \times k$$

Where $R_{\Theta}$ applies rotation matrices parameterized by frequencies:

$$\text{freqs}_i = \frac{1.0}{\text{base}^{\frac{2i}{d_{\text{head}}}}} \quad \text{for } i \in \left[0, \frac{d_{\text{head}}}{2}\right)$$

---

### KV Upscaler Backbone

The `KvHALO_Upscaler` manages the end-to-end reconstruction pipeline:

```
┌──────────────────────────────────────────────────────────────────┐
│                      KvHALO_Upscaler                             │
│                                                                  │
│  lossy_kv [B, S, 2 * teacher_dim]                                │
│       │                                                          │
│       ▼                                                          │
│  ┌──────────────┐   Linear(2 * teacher_dim → d_model)            │
│  │ Input        │   → z_L [B, S, d_model]                        │
│  │ Compression  │                                                │
│  └──────────────┘                                                │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────┐                                     │
│  │ HRM Thinking Loops (t=2) │   Iterative refinement             │
│  │  z_H, z_L ← HRMInner     │   Dual-stream information routing  │
│  └─────────────────────────┘                                     │
│       │                                                          │
│       ▼                                                          │
│  ┌──────────────┐   Linear(d_model → 2 * teacher_dim)            │
│  │ Regression   │   → predicted_states [B, S, 2 * teacher_dim]   │
│  │ Head         │                                                │
│  └──────────────┘                                                │
│       │                                                          │
│       ▼                                                          │
│  predicted_states [B, S, 2 * teacher_dim] ← Chunk → K_pred, V_pred  │
└──────────────────────────────────────────────────────────────────┘

```

#### Forward Pass Execution Flow:

1. **Input Compression:** The lossy, quantized KV states are projected from the full teacher space (`teacher_dim * 2`) down to a lower-dimensional latent space (`d_model`).
2. **HRM Thinking:** Exactly `t_steps=2` iterations of the dual-stream `HRMInner` architecture process the latent representations, progressively refining details.
3. **Regression:** The final high-state is projected back to the full teacher dimension, splitting out the predicted high-fidelity key and value states.

---

### Quantization & Reconstruction Pipeline

KVHALO provides uniform, configurable bit-width simulation mapping to $\text{Steps} = 2^{\text{bits}} - 1$:

* **2-bit:** 4 levels $[0, \frac{1}{3}, \frac{2}{3}, 1]$ $\rightarrow$ 8x spatial VRAM reduction
* **4-bit:** 15 levels $[0, \frac{1}{14}, \dots, 1]$ $\rightarrow$ 4x spatial VRAM reduction

The simulation normalizes tensors to a $[0, 1]$ range, rounds to the nearest discrete level, and scales values back to their original scalar magnitude.

---

## Project Structure

```
KVHALO-Inference/
├── main.py                           # Main production-ready client API 
├── README.md                         # Project documentation
├── LICENSE                           # Apache 2.0 License
├── requirements.txt                  # System dependencies
├── examples/
│   └── compare_base_mistral.py       # Interactive Gradio comparison UI
└── sub_libraries/
    ├── HRM_Main.py                   # HRM architecture & KvHALO_Upscaler definitions
    └── HRM_Simulated_Quantization.py # Numerical quantization utilities

```

### File Manifest

| File | Purpose |
| --- | --- |
| `main.py` | Exposes `MistralKvHALO`—the production API for seamlessly intercepting and patching models. |
| `sub_libraries/HRM_Simulated_Quantization.py` | Contains `simulate_low_bit_quantization()` for runtime/training emulation. |
| `examples/compare_base_mistral.py` | Runs a Gradio UI to compare baseline vs. KVHALO generations alongside structural charts. |

---

## Installation

### Prerequisites

* **Python 3.8+**
* **CUDA-compatible GPU** (highly recommended) or Apple Silicon Mac (with MPS enabled)
* **Hugging Face Account** (authenticated via CLI to download base weights)

### Setup

```bash
# Clone repository and enter directory
git clone https://github.com/RA86-dev/KVHALO_Inference.git
cd KVHALO_Inference

# Install requirements
pip install -r requirements.txt

# Download pre-trained upscaler weights
huggingface-cli download richyvd/kvhalo

```

---

## Usage

For seamless implementation, clone this repository directly inside your workspace root directory to reference local imports easily.

### Quick Start

```python
from KVHALO_Inference.main import MistralKvHALO

# Initialize the KVHALO patching engine
client = MistralKvHALO(target_layer=15)

# Load trained upscaler weights
client.load_upscaler_weights("best_KVHALO.pt")

# Stream generation with 2-bit target layers
prompt = "Explain quantum entanglement in simple terms."
for chunk in client.generate(prompt, max_new_tokens=64, bits=2, temperature=0.7):
    print(chunk, end="", flush=True)

```

### Programmatic API

For granular pipeline control, you can patch and unpatch target modules manually:

```python
from KVHALO_Inference.main import MistralKvHALO

client = MistralKvHALO(
    model_id="mistralai/Mistral-7B-Instruct-v0.3",
    target_layer=15
)
client.load_upscaler_weights("best_KVHALO.pt")

# Active runtime hook patching
client.patch(bits=2)

# ... run your custom generation loops or evaluation logic here ...

# Unpatching restores model back to original FP16 states
client.unpatch()

```

---

## Configuration

The model upscaler tracks configuration inputs matching standard distillation hyperparameters:

```python
CONFIG = {
    "d_model": 768,      # Latent representation dimension
    "n_heads": 12,       # Attention heads within the HRM block
    "d_ff": 3072,        # Hidden layer size of SwiGLU block (4 * d_model)
    "dropout": 0.0,      # Zeroed out for inference execution
    "teacher_dim": 1024  # Base model target width (Mistral: 8 GQA heads * 128 head_dim)
}

```

---

## Training

To train a custom `KvHALO_Upscaler` checkpoint:

1. **Collect Cache Pairs:** Extract corresponding pairs of original 16-bit states ("teacher") and target low-bit quantized states ("student") during forward passes.
2. **Optimize via Multi-Objective Loss:** Balance spatial geometry magnitude alongside directional semantics using a joint Mean Squared Error (MSE) and Cosine Similarity objective function:

$$L = \text{MSE}(\text{predicted}, \text{target}) + \big(1 - \text{CosineSimilarity}(\text{pred}_{\text{flat}}, \text{target}_{\text{flat}})\big)$$

3. **Export Checkpoint:**

```python
torch.save({
    "model_state_dict": upscaler.state_dict(),
    "config": CONFIG
}, "best_KVHALO.pt")

```

---

## Performance Metrics

### Memory Savings

On targets running long sequence context boundaries, VRAM constraints scale roughly as:

$$\text{VRAM Savings} \approx \text{Footprint}_{\text{FP16}} - \text{Footprint}_{\text{Quantized}}$$

* **Baseline FP16 layer cache (100 tokens):** ~128 KB
* **2-bit Quantized layer cache (100 tokens):** ~8 KB
* **Net Layer Savings:** ~120 KB per target layer

### Reconstruction Quality

* **Cosine Similarity Alignment:** Reaches **~91.85%** directional vector alignment relative to reference baseline states when running under 2-bit quantization metrics.
* **Linguistic Perplexity:** Substantially lowers the downstream linguistic degradation commonly found in naive, non-optimized quantizations.

### Execution Speeds

* **Warm-up Latency:** The initial forward token triggers a minor compilation pass (~2-5s on CUDA, ~5-10s on Apple Silicon MPS).
* **Per-Token Overhead:** Negligible addition of **~0.1–0.5ms per token** using the highly lightweight 92M parameter HRM architecture.

---

## Technical Details

### HRM Summary Specs

| Metric | Value |
| --- | --- |
| **Total Parameters** | ~46M |
| **Thinking Iterations ($t_{\text{steps}}$)** | 2 Loops |
| **State Layout** | Dual-Stream ($z_H$ Global + $z_L$ Residual) |
| **Attention Engine** | Multi-Head Attention via FlashAttention-2 |
| **Position Mapping** | Rotary Embeddings (RoPE), Base = 10,000 |

### Tested Deployments

* **Mistral-7B-Instruct-v0.3** (Grouped-Query Attention with 8 KV heads)
* *Architecture is design-agnostic—scale `teacher_dim` params upward to fit alternative GQA architectures.*

### Hardware Specs

| Requirements | Minimum | Recommended |
| --- | --- | --- |
| **GPU VRAM** | 8 GB | 16+ GB |
| **System RAM** | 16 GB | 32+ GB |
| **Apple Silicon** | M1 Engine or later | M1 Pro / Max or later |
| **CUDA Compute** | Capability 7.0+ | Capability 8.0+ |

---

## License and Paper
The paper is [located here](https://doi.org/10.5281/zenodo.20361517)
This framework is open-source software distributed under the terms of the **Apache License 2.0**.

```text
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
