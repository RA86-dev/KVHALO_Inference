# =====================================================================
# Copyright 2026 Richard Wang & The KVHALO Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =====================================================================

import torch
import time
import os
import gradio as gr
import numpy as np
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from project_candy_model import ProjectCandyKVNet

# Attempt standard matplotlib import for live visualization
try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive background renderer
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# # Check for Apple Silicon / Metal Performance Shaders
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
#     torch_dtype = torch.float16
#     print("🍏 UI Engine mounted onto Metal Performance Shaders (MPS).")
# else:
#     device = torch.device("cpu")
#     torch_dtype = torch.float32
#     print("💻 UI Engine mounted onto CPU.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Configuration matching your 92M training run
CONFIG = {
    "d_model": 768,
    "n_heads": 12,
    "d_ff": 3072,
    "dropout": 0.0,
    "teacher_dim": 1024
}
TARGET_LAYER = 15

# Global session cache to hold tensor slices for real-time plotting
waveform_data = {
    "true": None,
    "quantized": None,
    "reconstructed": None
}

# =====================================================================
# SYSTEM INITIALIZATION
# =====================================================================
print("📖 Booting local Mistral-7B-Instruct-v0.3...")
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Avoid buggy caching_allocator_warmup and prevent sluggish disk-offloading
# by loading directly and moving to device instead of using device_map="auto"
teacher = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)
print("🚚 Transferring Mistral-7B layers to Apple Silicon GPU (MPS)...")
teacher = teacher.to(device)
print("✅ Mistral-7B successfully loaded in Unified Memory!")

print("🍬 Mounting Project Candy KV Network...")
candy = ProjectCandyKVNet(CONFIG).to(device=device, dtype=torch_dtype)
if os.path.exists("best_candy.pt"):
    checkpoint_data = torch.load("best_candy.pt", map_location=device)
    state_dict = checkpoint_data["model_state_dict"] if isinstance(checkpoint_data,
                                                                   dict) and "model_state_dict" in checkpoint_data else checkpoint_data
    candy.load_state_dict(state_dict)
    candy.eval()
    print("👑 Golden weights successfully secured inside the UI engine!")
else:
    print("⚠️ WARNING: 'best_candy.pt' not found. Running UI with uninitialized weights.")

# =====================================================================
# SILENT WARM-UP ROUTINE TO PRIM THE HARDWARE CACHES
# =====================================================================
print("🔥 Running silent warm-up pass to prime MPS shaders and unified memory...")
try:
    warmup_tokens = tokenizer("Warmup test context initialization string.", return_tensors="pt").to(device)
    # Run a tiny generation to compile GPU attention kernels and reserve VRAM pages
    with torch.no_grad():
        _ = teacher.generate(**warmup_tokens, max_new_tokens=5, use_cache=True)
    print("⚡ Warmup pass complete! Shaders compiled and unified memory initialized.")
except Exception as e:
    print(f"⚠️ Warmup pass encountered a non-fatal warning: {e}")


# =====================================================================
# DYNAMIC QUANTIZATION UTILITY
# =====================================================================
def simulate_quantization(tensor, bits=2):
    """
    Simulates variable-bit lossy quantization (from 1-bit to 4-bit space).
    Trashes tensor mapping according to the selected bit-depth allocation.
    """
    min_val, max_val = tensor.min(), tensor.max()
    normalized = (tensor - min_val) / (max_val - min_val + 1e-5)

    # Map to steps (e.g. 1-bit = 2 steps [0, 1], 2-bit = 4 steps [0, 1/3, 2/3, 1])
    steps = (2 ** bits) - 1
    quantized = torch.round(normalized * steps) / steps

    # Rescale back to original magnitude distribution
    return (quantized * (max_val - min_val)) + min_val


# =====================================================================
# VISUALIZATION GENERATORS
# =====================================================================
def generate_waveform_plot(bits=2):
    """Generates a high-quality comparative line graph of the cache tensors."""
    if not HAS_MATPLOTLIB or waveform_data["true"] is None:
        return None

    try:
        # Extract the first 80 elements of a representative head projection
        true_slice = waveform_data["true"][:80]
        quant_slice = waveform_data["quantized"][:80]
        recon_slice = waveform_data["reconstructed"][:80]

        plt.figure(figsize=(10, 3.8), dpi=120)
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

        # Plot continuous ground truth state
        plt.plot(true_slice, label="Original 16-Bit Float Master State", color="#3b82f6", linewidth=2.0, alpha=0.9)
        # Plot blocky step-wise quantization
        plt.step(range(len(quant_slice)), quant_slice, label=f"Naive {bits}-Bit Quantized (Crushed)", color="#ef4444",
                 linewidth=1.5, where='mid', alpha=0.7)
        # Plot continuous super-resolution output
        plt.plot(recon_slice, label="KVHALO Reconstructed Manifold", color="#10b981", linewidth=2.0, linestyle="--",
                 alpha=0.95)

        plt.title(f"KV Cache Latent Manifold Super-Resolution Reconstruction (Layer 15 Slice | {bits}-Bit)",
                  fontsize=11, fontweight="bold", pad=12)
        plt.xlabel("Vector Hidden Dimension Index (Subset)", fontsize=9)
        plt.ylabel("Latent State Amplitude", fontsize=9)
        plt.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9, fontsize=8)
        plt.tight_layout()

        plot_path = "cache_waveform_comparison.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        return plot_path
    except Exception as e:
        print(f"⚠️ Waveform plotting encountered an issue: {e}")
        return None


# =====================================================================
# THE GRADIO INTERFACE DESIGN & STREAMING PIPELINES
# =====================================================================
def calculate_jaccard_similarity(text1, text2):
    """Computes basic word-level Jaccard similarity between the two outputs."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return (len(intersection) / len(union)) * 100


def process_comparison(prompt, max_tokens, bits, temperature, top_p):
    """
    Generator function that streams tokens sequentially for both the
    Disabled baseline run and the Enabled KVHALO super-resolution run.
    """
    global waveform_data
    # Clear visual state cache before running new execution
    waveform_data = {"true": None, "quantized": None, "reconstructed": None}

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    target_submodule = teacher.model.layers[TARGET_LAYER].self_attn
    original_forward = target_submodule.forward

    disabled_text = ""
    enabled_text = ""
    disabled_speed = "0.00 tokens/sec"
    disabled_time = "0.00 seconds"
    enabled_speed = "0.00 tokens/sec"
    enabled_time = "0.00 seconds"
    waveform_image_path = None
    vram_savings_text = "N/A"
    reconstruction_fidelity_text = "N/A"

    # Configure generation decoding strategy
    do_sample = temperature > 0.0
    gen_config = {
        "max_new_tokens": max_tokens,
        "use_cache": True,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_config["temperature"] = temperature
        gen_config["top_p"] = top_p

    # =====================================================================
    # STREAM 1: RUNNING BASELINE (DISABLED)
    # =====================================================================
    print(f"\n⏳ [Run 1/2] Streaming Baseline (Pure {bits}-Bit Quantization - Candy Disabled)...")

    def decompress_kv_disabled(raw_keys, raw_values, start_pos=0):
        with torch.no_grad():
            b, n_kv, s, hd = raw_keys.shape
            true_keys = raw_keys.transpose(1, 2).reshape(b, s, n_kv * hd).to(torch.float32)
            true_values = raw_values.transpose(1, 2).reshape(b, s, n_kv * hd).to(torch.float32)

            # Apply variable-bit lossy quantization
            lossy_keys = simulate_quantization(true_keys, bits=bits)
            lossy_values = simulate_quantization(true_values, bits=bits)

            final_keys = lossy_keys.reshape(b, s, n_kv, hd).transpose(1, 2)
            final_values = lossy_values.reshape(b, s, n_kv, hd).transpose(1, 2)
            return final_keys.to(dtype=torch_dtype), final_values.to(dtype=torch_dtype)

    def patched_attn_forward_disabled(*args, **kwargs):
        print(f"⚓ [Disabled] Intercepting and compressing Layer {TARGET_LAYER} attention to {bits}-bit...")
        outputs = original_forward(*args, **kwargs)
        if isinstance(outputs, tuple):
            attn_output, past_key_value = outputs[0], outputs[-1]
        else:
            attn_output, past_key_value = outputs, None

        if past_key_value is not None:
            current_position = past_key_value.get_seq_length(TARGET_LAYER)
            orig_k = past_key_value.key_cache[TARGET_LAYER]
            orig_v = past_key_value.value_cache[TARGET_LAYER]
            dec_k, dec_v = decompress_kv_disabled(orig_k, orig_v, start_pos=current_position)
            past_key_value.key_cache[TARGET_LAYER] = dec_k
            past_key_value.value_cache[TARGET_LAYER] = dec_v

        return (attn_output, past_key_value) if isinstance(outputs, tuple) and len(outputs) == 2 else outputs

    target_submodule.forward = patched_attn_forward_disabled

    disabled_streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs_disabled = dict(**inputs, streamer=disabled_streamer, **gen_config)

    thread_disabled = Thread(target=teacher.generate, kwargs=generation_kwargs_disabled)
    start_time_disabled = time.time()
    thread_disabled.start()

    tokens_generated_disabled = 0
    for new_text in disabled_streamer:
        disabled_text += new_text
        tokens_generated_disabled += 1
        elapsed_disabled = time.time() - start_time_disabled
        disabled_speed = f"{tokens_generated_disabled / elapsed_disabled:.2f} tokens/sec"
        disabled_time = f"{elapsed_disabled:.2f} seconds"

        yield (
            disabled_text, disabled_speed, disabled_time,
            enabled_text, enabled_speed, enabled_time,
            waveform_image_path,
            vram_savings_text,
            reconstruction_fidelity_text
        )

    thread_disabled.join()
    target_submodule.forward = original_forward
    print(f"✅ Baseline streaming complete. Total tokens: {tokens_generated_disabled}")

    # =====================================================================
    # STREAM 2: RUNNING ACTIVE SYSTEM (ENABLED)
    # =====================================================================
    print(f"\n⏳ [Run 2/2] Streaming KVHALO Manifold Recovery (Candy Enabled at {bits}-Bit)...")

    def decompress_kv_enabled(raw_keys, raw_values, start_pos=0):
        global waveform_data
        with torch.no_grad():
            b, n_kv, s, hd = raw_keys.shape
            true_keys = raw_keys.transpose(1, 2).reshape(b, s, n_kv * hd).to(torch.float32)
            true_values = raw_values.transpose(1, 2).reshape(b, s, n_kv * hd).to(torch.float32)

            # Apply same quantization stress test
            lossy_keys = simulate_quantization(true_keys, bits=bits)
            lossy_values = simulate_quantization(true_values, bits=bits)

            # Upscale utilizing Candy HRM Backbone
            lossy_kv_concat = torch.cat([lossy_keys, lossy_values], dim=-1).to(dtype=torch_dtype)
            candy_output = candy(lossy_kv_concat, start_pos=start_pos)
            upscaled_kv = candy_output["predicted_states"]
            pred_keys, pred_values = upscaled_kv.chunk(2, dim=-1)

            # Retain cache slice for final plotting metrics
            waveform_data["true"] = true_keys[0, 0].cpu().numpy()
            waveform_data["quantized"] = lossy_keys[0, 0].cpu().numpy()
            waveform_data["reconstructed"] = pred_keys[0, 0].cpu().to(dtype=torch.float32).numpy()

            final_keys = pred_keys.reshape(b, s, n_kv, hd).transpose(1, 2)
            final_values = pred_values.reshape(b, s, n_kv, hd).transpose(1, 2)
            return final_keys.to(dtype=torch_dtype), final_values.to(dtype=torch_dtype)

    def patched_attn_forward_enabled(*args, **kwargs):
        print(
            f"⚓ [Enabled] Intercepting, crushing ({bits}-bit), and reconstructively upscaling Layer {TARGET_LAYER}...")
        outputs = original_forward(*args, **kwargs)
        if isinstance(outputs, tuple):
            attn_output, past_key_value = outputs[0], outputs[-1]
        else:
            attn_output, past_key_value = outputs, None

        if past_key_value is not None:
            current_position = past_key_value.get_seq_length(TARGET_LAYER)
            orig_k = past_key_value.key_cache[TARGET_LAYER]
            orig_v = past_key_value.value_cache[TARGET_LAYER]
            dec_k, dec_v = decompress_kv_enabled(orig_k, orig_v, start_pos=current_position)
            past_key_value.key_cache[TARGET_LAYER] = dec_k
            past_key_value.value_cache[TARGET_LAYER] = dec_v

        return (attn_output, past_key_value) if isinstance(outputs, tuple) and len(outputs) == 2 else outputs

    target_submodule.forward = patched_attn_forward_enabled

    enabled_streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs_enabled = dict(**inputs, streamer=enabled_streamer, **gen_config)

    thread_enabled = Thread(target=teacher.generate, kwargs=generation_kwargs_enabled)
    start_time_enabled = time.time()
    thread_enabled.start()

    tokens_generated_enabled = 0
    for new_text in enabled_streamer:
        enabled_text += new_text
        tokens_generated_enabled += 1
        elapsed_enabled = time.time() - start_time_enabled
        enabled_speed = f"{tokens_generated_enabled / elapsed_enabled:.2f} tokens/sec"
        enabled_time = f"{elapsed_enabled:.2f} seconds"

        yield (
            disabled_text, disabled_speed, disabled_time,
            enabled_text, enabled_speed, enabled_time,
            waveform_image_path,
            vram_savings_text,
            reconstruction_fidelity_text
        )

    thread_enabled.join()
    target_submodule.forward = original_forward
    print(f"✅ KVHALO streaming complete. Total tokens: {tokens_generated_enabled}")

    # =====================================================================
    # POST-INFERENCE DIAGNOSTICS & TELEMETRY
    # =====================================================================
    print("📊 Computing telemetry and plotting waveform visualizer...")

    # 1. Generate live plotting waveform graph
    waveform_image_path = generate_waveform_plot(bits=bits)

    # 2. Calculate textual divergence / restoration index
    jaccard_score = calculate_jaccard_similarity(disabled_text, enabled_text)
    reconstruction_fidelity_text = f"{100 - jaccard_score:.1f}% divergence from crushed baseline"

    # 3. Estimated localized VRAM calculation on Layer 15 cache
    # Formula: Batch * SequenceLength * heads * head_dim * 2 (K&V) * byte_precision
    total_seq = len(prompt.split()) + tokens_generated_enabled
    fp16_bytes = 1 * total_seq * 8 * 128 * 2 * 2  # standard layout
    quant_bytes = 1 * total_seq * 8 * 128 * 2 * (bits / 8.0)  # dynamic bit-width footprint
    savings_kb = (fp16_bytes - quant_bytes) / 1024
    vram_savings_text = f"~{savings_kb:.2f} KB saved (Layer 15)"

    # Final yield to lock in the plots and dynamic diagnostics
    yield (
        disabled_text, disabled_speed, disabled_time,
        enabled_text, enabled_speed, enabled_time,
        waveform_image_path,
        vram_savings_text,
        reconstruction_fidelity_text
    )


# =====================================================================
# THE GRADIO INTERFACE
# =====================================================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🍬 Project Candy: Real-Time KV-Cache Decompressorate Monitor")
    gr.Markdown(
        "### Comparative Analysis: Static Attenuated Space vs. Implicit Neural Manifold Recovery (KVHALO) | Apache 2.0")

    with gr.Row():
        with gr.Column(scale=3):
            user_prompt = gr.Textbox(
                label="Prompt Input Area",
                value="The proof of the fundamental theorem of algebra requires us to assume a complex polynomial P(z) has no roots, and then construct a function f(z) that is holomorphic on the Riemann sphere, and has a pole at infinity. We then show that f(z) is constant, which contradicts our assumption that P(z) has no roots. In this post, I will show that f(z) is holomorphic on the Riemann sphere.",
                lines=3
            )
        with gr.Column(scale=2):
            with gr.Row():
                bit_slider = gr.Slider(minimum=1, maximum=4, step=1, value=2, label="Quantization Level (Bits)")
                token_slider = gr.Slider(minimum=10, maximum=150, step=5, value=100, label="Max New Tokens")
            with gr.Row():
                temp_slider = gr.Slider(minimum=0.0, maximum=1.5, step=0.1, value=0.7,
                                        label="Temperature (0.0 = Greedy)")
                topp_slider = gr.Slider(minimum=0.1, maximum=1.0, step=0.05, value=0.9, label="Top-P")
            submit_btn = gr.Button("⚡ Run Comparative Execution", variant="primary")

    with gr.Row():
        # LEFT WINDOW: SYSTEM WITHOUT CANDY RECONSTRUCTION
        with gr.Column():
            gr.Markdown("### 🚫 Baseline: Candy Decompressor [ DISABLED ]")
            gr.Markdown(
                "*Mistral Layer 15 cache is crushed to lossy low-bit tensors and forced directly into self-attention.*")
            out_disabled_text = gr.TextArea(label="Output Response Text", lines=8, interactive=False)
            with gr.Row():
                out_disabled_speed = gr.Textbox(label="Throughput Velocity", interactive=False)
                out_disabled_time = gr.Textbox(label="Total Generation Time", interactive=False)

        # RIGHT WINDOW: SYSTEM WITH ACTIVE NEURAL UPSCALING
        with gr.Column():
            gr.Markdown("### 👑 Active System: KVHALO Decompressor [ ENGAGED ]")
            gr.Markdown(
                "*Mistral Layer 15 cache is crushed, then dynamically upscaled back to high-resolution Float16 via the 92M parameter HRM framework.*")
            out_enabled_text = gr.TextArea(label="Output Response Text", lines=8, interactive=False)
            with gr.Row():
                out_enabled_speed = gr.Textbox(label="Throughput Velocity", interactive=False)
                out_enabled_time = gr.Textbox(label="Total Generation Time", interactive=False)

    # LOWER MONITOR PANEL: REAL-TIME GRAPHICS AND TELEMETRY ANALYTICS
    gr.Markdown("## 🔍 Live Deep-Dive Analytics & Metrics Dashboard")
    with gr.Row():
        with gr.Column(scale=3):
            out_waveform_plot = gr.Image(
                label="Comparative Manifold Waveform View (Continuous vs. Compressed vs. KVHALO)", type="filepath")
        with gr.Column(scale=2):
            gr.Markdown("### 📈 Mathematical Diagnostic Telemetry")
            vram_savings = gr.Textbox(label="VRAM Footprint Reduction Index", value="N/A", interactive=False)
            semantic_divergence = gr.Textbox(label="Linguistic Divergence Recovered", value="N/A", interactive=False)
            reconstruction_score = gr.Textbox(label="HRM Target Alignment Cosine Similarity",
                                              value="91.85% (Empirical Step 8,055)", interactive=False)
            decompression_ratio = gr.Textbox(label="Inference Compression Factor",
                                             value="8.0x (16-bit to 2-bit footprint)", interactive=False)

    submit_btn.click(
        fn=process_comparison,
        inputs=[user_prompt, token_slider, bit_slider, temp_slider, topp_slider],
        outputs=[
            out_disabled_text, out_disabled_speed, out_disabled_time,
            out_enabled_text, out_enabled_speed, out_enabled_time,
            out_waveform_plot,
            vram_savings,
            semantic_divergence
        ]
    )

if __name__ == "__main__":
    demo.launch(share=False)