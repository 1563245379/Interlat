# """
# Aligned Hidden Injection – v2.1
# ===============================
# Drop‑in replacement for `process_hidden_states` with richer control:
#     • (optional)  Self‑MHA  – hidden↔hidden pre‑mix
#     • Cross‑Attention      – hidden (Q) × context (K/V)
#     • Gated Adapter        – sample‑wise injection strength
#     • Conditional LoRA     – context‑conditioned low‑rank projection
#     • Context LayerNorm    – keep K/V scale close to hidden
#     • save_attn flag       – extract attention heat‑map for viz

# **New Args**
# ------------
# AlignBlock(
#     d_model: int,
#     n_heads: int = 8,
#     lora_rank: int = 4,
#     use_self_mha: bool = False,
#     norm_ctx: bool = True,
# )

# HOW TO USE
# ~~~~~~~~~~
# ```python
# self.align_block = AlignBlock(
#     d_model=self.hidden_size,
#     n_heads=num_heads,
#     lora_rank=4,
#     use_self_mha=False,      # turn on if hidden has internal structure
# )
# ...
# aligned, attn = self.align_block(hidden_states, context_embeds, save_attn=True)
# # attn : (B,H,L_h,L_ctx)  – averaged per head if save_attn=False
# ```
# """

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. Conditional LoRA – generate (A,B) from context
# ---------------------------------------------------------------------------
class CondLoRA(nn.Module):
    """Low‑rank adapter: produce rank‑r (A,B) conditioned on a context vector."""
    def __init__(self, d_model: int, rank: int = 4, hidden_factor: int = 4):
        super().__init__()
        self.d, self.r = d_model, rank
        hidden = hidden_factor * d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, 2 * d_model * rank)
        )
        self.scale = 1 / math.sqrt(rank)

    def forward(self, ctx: torch.Tensor):
        # ctx : (B,D)
        B, r, d = ctx.size(0), self.r, self.d
        params = self.mlp(ctx)                   # (B, 2*d*r)
        A, Bm = params.chunk(2, -1)
        A  = A.view(B, r, d) * self.scale        # (B,r,d)
        Bm = Bm.view(B, r, d) * self.scale
        return A, Bm

# ---------------------------------------------------------------------------
# 2. LoRA‑enhanced projection
# ---------------------------------------------------------------------------
class CondLoRAProj(nn.Module):
    def __init__(self, d_model: int, rank: int = 8):
        super().__init__()
        self.base = nn.Linear(d_model, d_model)
        self.lora = CondLoRA(d_model, rank) if rank > 0 else None
        self.drop = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, ctx_pool: torch.Tensor):
        out = self.base(x)
        if self.lora is not None:
            A, B = self.lora(ctx_pool)           # (B,r,d)
            # x: (B,L,d)  –> (B,L,r) –> (B,L,d)
            lora_out = torch.einsum('blr,brd->bld', torch.einsum('bld,brd->blr', x, B), A)
            out = out + lora_out
        return self.drop(out)

# ---------------------------------------------------------------------------
# 3. Gated Adapter (scalar gate)
# ---------------------------------------------------------------------------
class GatedAdapter(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)
        nn.init.constant_(self.linear.bias, 0.5)

    def forward(self, ctx_pool: torch.Tensor) -> torch.Tensor:  # (B,D) -> (B,1)
        return torch.sigmoid(self.linear(ctx_pool))

# # ---------------------------------------------------------------------------
# # 4. AlignBlock end‑to‑end
# # ---------------------------------------------------------------------------
# class AlignBlock(nn.Module):
#     def __init__(self, d_model: int, n_heads: int = 8, lora_rank: int = 4,
#                  use_self_mha: bool = False, norm_ctx: bool = True, use_gate: bool = False):
#         super().__init__()

#         self.use_self_mha = use_self_mha
#         self.norm_ctx = norm_ctx
#         if use_self_mha:
#             self.self_mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.1)
#             self.self_ln  = nn.LayerNorm(d_model)
#             self._init_mha(self.self_mha)

#         self.ca = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.1)
#         self.ca_ln = nn.LayerNorm(d_model)
#         self._init_mha(self.ca)

#         if norm_ctx:
#             self.ctx_ln = nn.LayerNorm(d_model)

#         self.gate = GatedAdapter(d_model)
#         self.proj = CondLoRAProj(d_model, rank=lora_rank)
#         self.proj_ln = nn.LayerNorm(d_model)

#         # Learnable γ to down‑scale LN output if needed
#         self.gamma = nn.Parameter(torch.tensor(0.1))

#         # cache for viz
#         self.last_attn: torch.Tensor | None = None
#         self.last_gate: torch.Tensor | None = None

#     @staticmethod
#     def _init_mha(mha: nn.MultiheadAttention):
#         nn.init.xavier_uniform_(mha.in_proj_weight, gain=1.0 / math.sqrt(3))
#         nn.init.xavier_uniform_(mha.out_proj.weight, gain=1.0)
#         if mha.in_proj_bias is not None:
#             nn.init.zeros_(mha.in_proj_bias)
#         if mha.out_proj.bias is not None:
#             nn.init.zeros_(mha.out_proj.bias)

#     def forward(self, hidden: torch.Tensor, context: torch.Tensor, *, save_attn=False):
#         # hidden  : (B,L_h,D)
#         # context : (B,L_c,D)
#         if self.norm_ctx:
#             context = self.ctx_ln(context)

#         # --- optional Self‑MHA (hidden ↔ hidden) ---
#         if self.use_self_mha:
#             h_self, _ = self.self_mha(hidden, hidden, hidden)
#             hidden = self.self_ln(hidden + h_self)

#         # --- Cross‑Attention (hidden Q , context K/V) ---
#         ca_out, attn_w = self.ca(hidden, context, context,
#                                  need_weights=save_attn,
#                                  average_attn_weights=False)
#         hidden = self.ca_ln(hidden + ca_out)
#         if save_attn:
#             # store averaged over batch for viz (H,L_h,L_c)
#             self.last_attn = attn_w.mean(0).detach().cpu()

#         # --- Gate ---
#         ctx_pool = context.mean(dim=1)
#         g = self.gate(ctx_pool)              # (B,1)
#         self.last_gate = g.detach().cpu()

#         # --- Conditional LoRA Projection ---
#         proj_out = self.proj(hidden, ctx_pool)  # (B,L_h,D)
#         out = g.unsqueeze(1) * proj_out + (1.0 - g.unsqueeze(1)) * hidden
#         out = self.proj_ln(out) * self.gamma
#         return out, attn_w if save_attn else None

class AlignBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, lora_rank: int = 4,
                 use_self_mha: bool = False, norm_ctx: bool = True,
                 use_gate: bool = False):
        super().__init__()

        self.use_self_mha = use_self_mha
        self.norm_ctx = norm_ctx
        self.use_gate = use_gate  # 添加这个属性

        if use_self_mha:
            self.self_mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.1)
            self.self_ln = nn.LayerNorm(d_model)
            self._init_mha(self.self_mha)

        self.ca = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.1)
        self.ca_ln = nn.LayerNorm(d_model)
        self._init_mha(self.ca)

        if norm_ctx:
            self.ctx_ln = nn.LayerNorm(d_model)

        # 只在use_gate=True时创建gate
        if use_gate:
            self.gate = GatedAdapter(d_model)

        self.proj = CondLoRAProj(d_model, rank=lora_rank)
        self.proj_ln = nn.LayerNorm(d_model)

        # Learnable γ to down‑scale LN output if needed
        self.gamma = nn.Parameter(torch.tensor(0.5))

        # cache for viz
        self.last_attn: torch.Tensor | None = None
        self.last_gate: torch.Tensor | None = None

    @staticmethod
    def _init_mha(mha: nn.MultiheadAttention):
        nn.init.xavier_uniform_(mha.in_proj_weight, gain=1.0 / math.sqrt(3))
        nn.init.xavier_uniform_(mha.out_proj.weight, gain=1.0)
        if mha.in_proj_bias is not None:
            nn.init.zeros_(mha.in_proj_bias)
        if mha.out_proj.bias is not None:
            nn.init.zeros_(mha.out_proj.bias)

    def forward(self, hidden: torch.Tensor, context: torch.Tensor, *, save_attn=False):
        # hidden  : (B,L_h,D)
        # context : (B,L_c,D)
        print("context device:", context.device)
        print("ctx_ln device:", next(self.ctx_ln.parameters()).device)

        if self.norm_ctx:
            context = self.ctx_ln(context)

        # --- optional Self‑MHA (hidden ↔ hidden) ---
        if self.use_self_mha:
            h_self, _ = self.self_mha(hidden, hidden, hidden)
            hidden = self.self_ln(hidden + h_self)

        # --- Cross‑Attention (hidden Q , context K/V) ---
        ca_out, attn_w = self.ca(hidden, context, context,
                                 need_weights=save_attn,
                                 average_attn_weights=False)
        # ca_out, attn_w = self.ca(context, hidden, hidden,
        #                          need_weights=save_attn,
        #                          average_attn_weights=False)
        hidden = self.ca_ln(hidden + ca_out)
        if save_attn:
            # store averaged over batch for viz (H,L_h,L_c)
            self.last_attn = attn_w.mean(0).detach().cpu()

        # --- Conditional LoRA Projection ---
        ctx_pool = context.mean(dim=1)
        proj_out = self.proj(hidden, ctx_pool)  # (B,L_h,D)

        # --- Gate (only if use_gate=True) ---
        if self.use_gate:
            g = self.gate(ctx_pool)  # (B,1)
            self.last_gate = g.detach().cpu()
            # 使用gate进行混合
            out = g.unsqueeze(1) * proj_out + (1.0 - g.unsqueeze(1)) * hidden
        else:
            # 不使用gate时的处理方式
            # 选项1: 直接使用projection输出
            # out = proj_out

            # 选项2: 固定比例混合 (例如0.5的权重)
            # out = 0.5 * proj_out + 0.5 * hidden

            # 选项3: 简单相加
            out = hidden + proj_out

            # 选项4: 使用residual connection
            # out = hidden + 0.1 * proj_out

            self.last_gate = None  # 没有gate时设为None

        out = self.proj_ln(out) * self.gamma
        return out, attn_w if save_attn else None
