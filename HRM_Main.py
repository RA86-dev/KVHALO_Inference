import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, d_head, block_size=8192, base=10000):
        super().__init__()
        self.d_head = d_head
        freqs = 1.0 / (base ** (torch.arange(0, self.d_head, 2).float() / self.d_head))
        self.register_buffer("freqs", freqs)
        pos_ids = torch.arange(block_size).float()
        angles = torch.outer(pos_ids, freqs)
        self.register_buffer("angles_cos", torch.cos(angles))
        self.register_buffer("angles_sin", torch.sin(angles))

    def apply(self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, n_heads, seq_len, d_head = q.shape
        q1, q2 = q[..., :d_head // 2], q[..., d_head // 2:]
        k1, k2 = k[..., :d_head // 2], k[..., d_head // 2:]
        cos = self.angles_cos[start_pos : start_pos + seq_len, :].view(1, 1, seq_len, d_head // 2)
        sin = self.angles_sin[start_pos : start_pos + seq_len, :].view(1, 1, seq_len, d_head // 2)
        q_rot = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
        k_rot = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
        return q_rot, k_rot

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5): 
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        return self.weight * (x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps))

class SwiGLUMuchPelu(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))

class HRMBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, rope_base=10000):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model
        
        self.norm1 = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLUMuchPelu(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbeddings(d_head=self.d_head, base=rope_base)
        
    def forward(self, x, start_pos=0):
        batch_size, seq_len, _ = x.shape
        x_norm = self.norm1(x)
        
        q = self.q_proj(x_norm).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x_norm).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x_norm).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        q, k = self.rope.apply(q, k, start_pos=start_pos)
        
        # PyTorch 2.0 Optimization: FlashAttention + Automatic Causal Masking
        is_causal = (seq_len > 1) 
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attn_out = self.o_proj(attn_out)
        
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

class HRMInner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.H_module = HRMBlock(config["d_model"], config["n_heads"], config["d_ff"], config["dropout"])
        self.L_module = HRMBlock(config["d_model"], config["n_heads"], config["d_ff"], config["dropout"])
        self.loop_norm = RMSNorm(config["d_model"]) 
        
    def forward(self, z_H, z_L, start_pos=0):
        # The Dual-Stream Information Routing (High/Low State Interplay)
        z_L_input = self.loop_norm(z_L + z_H) 
        z_L_current = self.L_module(z_L_input, start_pos=start_pos)
        z_H_input = z_H + z_L_current
        z_H_new = self.H_module(z_H_input, start_pos=start_pos)
        return z_H_new, z_L_current

# ---------------------------------------------------------
# THE KV UPSCALER BACKBONE (SUPER-RESOLUTION REGRESSOR)
# ---------------------------------------------------------
class KvHALO_Upscaler(nn.Module):
    def __init__(self, config, compiled=False):
        super().__init__()
        self.config = config
        self.compiled = compiled
        self.teacher_dim = config.get("teacher_dim", 4096)
        
        # Ingests the concatenated, quantized 2-bit Keys and Values
        # Shape goes from (4096 * 2) -> 768 bottleneck
        self.input_compression = nn.Linear(self.teacher_dim * 2, config["d_model"], bias=False)
        self.inner_model = HRMInner(config)
        
        # HARDCODED COMPUTE LIFECYCLE (t_steps = 2)
        self.t_steps = 2
        
        # CONTINUOUS REGRESSION HEAD: Outputs continuous f32 vectors matching Teacher KV
        # Shape blows back up from 768 -> (4096 * 2)
        self.regression_head = nn.Linear(config["d_model"], self.teacher_dim * 2, bias=False)

    def forward(self, lossy_kv_states, target_states=None, start_pos=0) -> Dict[str, torch.Tensor]:
        """
        lossy_kv_states: [batch_size, seq_len, teacher_dim * 2] (The corrupted 2-bit footprint)
        target_states:   [batch_size, seq_len, teacher_dim * 2] (The perfect 16-bit target)
        """
        batch_size, seq_len, _ = lossy_kv_states.shape

        # Step 0: Compress lossy KV cache down to candidate latent space
        z_L = self.input_compression(lossy_kv_states)
        z_H = torch.zeros_like(z_L)

        # STEP 1: Execute exactly 2 thinking loops through the dual-stream HRM architecture
        for step in range(self.t_steps):
            z_H, z_L = self.inner_model(z_H, z_L, start_pos=start_pos)
        
        # STEP 2: Project final latent state into the continuous teacher dimension
        # Shape: [batch_size, seq_len, teacher_dim * 2]
        predicted_f32_states = self.regression_head(z_H)
        
        output = {"predicted_states": predicted_f32_states}

        # STEP 3: Continuous Latent Distillation Loss Engine
        if target_states is not None:
            # FORCE TARGET STATES TO FLOAT32 TO PREVENT RUNTIME ERROR
            target_states_f32 = target_states.to(dtype=torch.float32)
            
            # A. Mean Squared Error (Geometric magnitude alignment)
            mse_loss = F.mse_loss(predicted_f32_states, target_states_f32)
            
            # B. Cosine Distance Loss (Directional semantic alignment)
            # We flatten everything to directly compare the pointing angles of the KV vectors
            flat_pred = predicted_f32_states.view(-1, self.teacher_dim * 2)
            flat_target = target_states_f32.view(-1, self.teacher_dim * 2)
            
            cosine_sim = F.cosine_similarity(flat_pred, flat_target, dim=-1)
            cosine_loss = 1.0 - cosine_sim.mean()
            
            # Balanced loss combination
            output["loss"] = mse_loss + cosine_loss
            output["mse_loss"] = mse_loss
            output["cosine_similarity"] = cosine_sim.mean()

        return output

    def compile(self):
        if not self.compiled:
            torch.compile(self)
            self.compiled = True
            print("🍬 ProjectCandyKVNet compiled successfully.")
        return self