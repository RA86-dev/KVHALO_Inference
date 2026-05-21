import torch
import torch.nn as nn
from typing import Dict, Any, Generator
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from sub_libraries.HierarchicalReasoningArchitecture import KvHALO_Upscaler

class MistralKvHALO:
    """
    Client wrapper for running Mistral-7B with token-level conditional 
    KV-cache compression and neural manifold reconstruction via KvHALO.
    """
    def __init__(self, model_id: str = "mistralai/Mistral-7B-Instruct-v0.3", target_layer: int = 15):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.torch.mps.is_available() else "cpu")
        self.torch_dtype = torch.float16 if self.device.type != "cpu" else torch.float32
        self.target_layer = target_layer
        
        print(f"📖 Loading tokenizer and base teacher model: {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=self.torch_dtype
        ).to(self.device)
        
        # Default 92M parameter configuration matching your training runs
        self.config = {
            "d_model": 768,
            "n_heads": 12,
            "d_ff": 3072,
            "dropout": 0.0,
            "teacher_dim": 1024  # 8 heads * 128 head_dim for Mistral GQA KV
        }
        
        print("🍬 Initializing KvHALO Neural Engine...")
        self.upscaler = KvHALO_Upscaler(self.config).to(device=self.device, dtype=self.torch_dtype)
        self.upscaler.eval()
        
        # Save reference to original forward pass to allow clean unpatching
        self.target_submodule = self.base_model.model.layers[self.target_layer].self_attn
        self.original_forward = self.target_submodule.forward
        self._is_patched = False

    def load_upscaler_weights(self, checkpoint_path: str):
        """Loads fine-tuned continuous regression weights into the HRM engine."""
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location=self.device)
            state_dict = checkpoint_data["model_state_dict"] if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data else checkpoint_data
            self.upscaler.load_states(state_dict) if hasattr(self.upscaler, 'load_states') else self.upscaler.load_state_dict(state_dict)
            print(f"👑 Secured golden weights from: {checkpoint_path}")
        except FileNotFoundError:
            print(f"⚠️ Warning: Checkpoint '{checkpoint_path}' not found. Operating with uninitialized weights.")

    def _simulate_quantization(self, tensor: torch.Tensor, bits: int) -> torch.Tensor:
        """Simulates discrete variable-bit lossy quantization footprint."""
        min_val, max_val = tensor.min(), tensor.max()
        normalized = (tensor - min_val) / (max_val - min_val + 1e-5)
        steps = (2 ** bits) - 1
        quantized = torch.round(normalized * steps) / steps
        return (quantized * (max_val - min_val)) + min_val

    def _build_patched_forward(self, bits: int):
        """Generates a dynamic attention patch tailored to the requested bit depth."""
        def patched_forward(*args, **kwargs):
            outputs = self.original_forward(*args, **kwargs)
            
            if isinstance(outputs, tuple):
                attn_output, past_key_value = outputs[0], outputs[-1]
            else:
                attn_output, past_key_value = outputs, None

            if past_key_value is not None:
                # Determine total sequence length in cache
                full_seq_len = past_key_value.get_seq_length(self.target_layer)
                
                # Check if we are in Prefill (many tokens) vs Decoding (1 new token)
                # We fetch ONLY the newly added token slices from the tail of the cache arrays
                is_decoding = kwargs.get("position_ids") is not None and kwargs["position_ids"].shape[-1] == 1
                slice_len = 1 if is_decoding else full_seq_len
                start_pos = full_seq_len - slice_len

                with torch.no_grad():
                    # Isolate active token slices
                    orig_k = past_key_value.key_cache[self.target_layer][..., -slice_len:, :]
                    orig_v = past_key_value.value_cache[self.target_layer][..., -slice_len:, :]
                    
                    b, n_kv, s, hd = orig_k.shape
                    
                    # Flatten heads to shape expected by HRM [B, S, Teacher_Dim]
                    true_keys = orig_k.transpose(1, 2).reshape(b, s, n_kv * hd).to(torch.float32)
                    true_values = orig_v.transpose(1, 2).reshape(b, s, n_kv * hd).to(torch.float32)

                    # Compress active token representations
                    lossy_keys = self._simulate_quantization(true_keys, bits=bits)
                    lossy_values = self._simulate_quantization(true_values, bits=bits)

                    # Reconstruct high-fidelity manifold via KvHALO
                    lossy_kv_concat = torch.cat([lossy_keys, lossy_values], dim=-1).to(dtype=self.torch_dtype)
                    kvhalo_output = self.upscaler(lossy_kv_concat, start_pos=start_pos)
                    pred_keys, pred_values = kvhalo_output["predicted_states"].chunk(2, dim=-1)

                    # Format back to Mistral head-layout
                    dec_k = pred_keys.reshape(b, s, n_kv, hd).transpose(1, 2).to(dtype=self.torch_dtype)
                    dec_v = pred_values.reshape(b, s, n_kv, hd).transpose(1, 2).to(dtype=self.torch_dtype)

                    # Overwrite tail elements of cache structure with upscaled states
                    past_key_value.key_cache[self.target_layer][..., -slice_len:, :] = dec_k
                    past_key_value.value_cache[self.target_layer][..., -slice_len:, :] = dec_v

            return (attn_output, past_key_value) if isinstance(outputs, tuple) and len(outputs) == 2 else outputs
            
        return patched_forward

    def patch(self, bits: int = 2):
        """Engages the KvHALO runtime hook onto Layer Attention structures."""
        if not self._is_patched:
            self.target_submodule.forward = self._build_patched_forward(bits=bits)
            self._is_patched = True
            print(f"⚓ KvHALO hooked onto Layer {self.target_layer} Attention (Target: {bits}-bit compression).")

    def unpatch(self):
        """Restores the standard, unaltered Mistral attention execution path."""
        if self._is_patched:
            self.target_submodule.forward = self.original_forward
            self._is_patched = False
            print(f"🔄 Layer {self.target_layer} Attention restored to vanilla 16-bit float pass.")

    def generate(self, prompt: str, max_new_tokens: int = 64, bits: int = 2, temperature: float = 0.7) -> Generator[str, None, None]:
        """
        Runs generation streaming tokens back interactively. Automatically manages 
        dynamic token-by-token neural recovery hooks during lifecycle.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        gen_config = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "do_sample": temperature > 0.0,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        if temperature > 0.0:
            gen_config["temperature"] = temperature

        # Inject the context-aware compression hooks
        self.patch(bits=bits)
        
        try:
            thread = Thread(target=self.base_model.generate, kwargs=gen_config)
            thread.start()
            
            for token_text in streamer:
                yield token_text
                
            thread.join()
        finally:
            # Always cleanly restore attention paths if generation faults or interrupts
            self.unpatch()

# ---------------------------------------------------------
# HOW TO EXECUTE YOUR PIPELINE CLEANLY:
# ---------------------------------------------------------
if __name__ == "__main__":
    # Initialize the engine client
    client = MistralKvHALO(target_layer=15)
    
    # Secure trained network patterns
    client.load_upscaler_weights("best_KVHALO.pt")
    
    context = "Explain the fundamental difference between Riemann integration and Lebesgue integration."
    print(f"\nPrompt: {context}\n")
    print("--- Streaming Active Engine Response ---")
    
    # Process generation with strict token-level conditional execution
    for response_chunk in client.generate(context, max_new_tokens=80, bits=2, temperature=0.5):
        print(response_chunk, end="", flush=True)
    print("\n-----------------------------------------")