from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask


class GPTBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


@dataclass
class GPTConfig:
    vocab_size: int
    max_seq_len: int
    embed_dim: int = 512
    num_layers: int = 8
    num_heads: int = 8
    dropout: float = 0.1


class GPTNextFrame(nn.Module):
    """GPT-like decoder that models P(next_frame_tokens | prev_frame_tokens) using a single causal stream.

    Training: Feed sequence [BOS, prev..., SEP, next...]; compute loss only on next tokens positions.
    Inference: Condition on [BOS, prev..., SEP] and autoregressively sample next tokens.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.embed_dim)
        self.seg_embed = nn.Embedding(2, cfg.embed_dim)  # 0=prev, 1=next
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([GPTBlock(cfg.embed_dim, cfg.num_heads, dropout=cfg.dropout) for _ in range(cfg.num_layers)])
        self.ln_f = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        b, t = input_ids.shape
        device = input_ids.device
        if t > self.cfg.max_seq_len:
            raise ValueError(f"Sequence too long: {t} > max_seq_len={self.cfg.max_seq_len}")
        pos_ids = torch.arange(t, device=device).unsqueeze(0).expand(b, -1)
        x = self.token_embed(input_ids) + self.pos_embed(pos_ids) + self.seg_embed(segment_ids)
        x = self.drop(x)
        attn_mask = build_causal_mask(t, device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        prefix_ids: torch.Tensor,
        prefix_segs: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        forbidden_token_ids: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        self.eval()
        generated = prefix_ids
        generated_segs = prefix_segs
        device = prefix_ids.device
        for _ in range(max_new_tokens):
            logits = self.forward(generated, generated_segs)
            next_token_logits = logits[:, -1, :] / max(1e-6, temperature)
            if forbidden_token_ids is not None:
                next_token_logits[:, list(forbidden_token_ids)] = float("-inf")
            if top_k is not None and top_k > 0:
                topk_vals, topk_idx = torch.topk(next_token_logits, k=min(top_k, next_token_logits.shape[-1]))
                filtered = torch.full_like(next_token_logits, float("-inf"))
                filtered.scatter_(1, topk_idx, topk_vals)
                next_token_logits = filtered
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_seg = torch.ones_like(next_token)  # next tokens segment id = 1
            generated = torch.cat([generated, next_token], dim=1)
            generated_segs = torch.cat([generated_segs, next_seg], dim=1)
        return generated


