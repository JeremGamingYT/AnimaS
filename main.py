"""
Script principal pour ANIMA‑S.

Ce script combine les modèles de l'auto‑encodeur, du GNN et du modèle séquentiel
pour illustrer l'utilisation de la chaîne complète.  Il permet de :

1. Charger des frames PNG et les reconstruire via l'auto‑encodeur.
2. Extraire les graphes sémantiques des images et appliquer des modifications
   simples avec le GNN.
3. (Optionnel) Prédire des graphes futurs à l'aide du modèle séquentiel et
   reconstruire les images correspondantes.

En raison de la complexité du pipeline complet, ce script fournit une
démo simplifiée et doit être adapté pour un véritable usage de production.
"""
import argparse
import math
import os
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm import tqdm
import random


# ------------------------------- Dataset ------------------------------------

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


def default_transforms(image_size: int, is_train: bool) -> Callable[[Image.Image], torch.Tensor]:
    if is_train:
        return T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                T.CenterCrop(image_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
    else:
        return T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )


def apply_pair_transforms(img_a: Image.Image, img_b: Image.Image, image_size: int, is_train: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    # Apply the same random choices to both images to preserve temporal alignment
    resize_crop = T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop(image_size),
    ])
    img_a = resize_crop(img_a)
    img_b = resize_crop(img_b)
    if is_train:
        if random.random() < 0.5:
            img_a = TF.hflip(img_a)
            img_b = TF.hflip(img_b)
    to_tensor = T.ToTensor()
    norm = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    xa = norm(to_tensor(img_a))
    xb = norm(to_tensor(img_b))
    return xa, xb


class FramePairDataset(Dataset):
    def __init__(
        self,
        root: str,
        image_size: int = 128,
        is_train: bool = True,
        sequence_by_subfolder: bool = False,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.image_size = int(image_size)
        self.is_train = bool(is_train)
        self.sequence_by_subfolder = bool(sequence_by_subfolder)
        self.transform = transform or default_transforms(self.image_size, is_train)

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        self.pairs: List[Tuple[Path, Path]] = []
        if self.sequence_by_subfolder:
            for sub in sorted([p for p in self.root.iterdir() if p.is_dir()]):
                frames = sorted([p for p in sub.iterdir() if p.is_file() and _is_image(p)])
                for i in range(len(frames) - 1):
                    self.pairs.append((frames[i], frames[i + 1]))
        else:
            frames = sorted([p for p in self.root.iterdir() if p.is_file() and _is_image(p)])
            for i in range(len(frames) - 1):
                self.pairs.append((frames[i], frames[i + 1]))

        if len(self.pairs) == 0:
            raise RuntimeError(f"No consecutive pairs found in: {self.root}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path_t, path_tp1 = self.pairs[index]
        img_t = Image.open(path_t).convert("RGB")
        img_tp1 = Image.open(path_tp1).convert("RGB")
        x_t, x_tp1 = apply_pair_transforms(img_t, img_tp1, self.image_size, self.is_train)
        return x_t, x_tp1


class SingleImageDataset(Dataset):
    def __init__(
        self,
        root: str,
        image_size: int = 128,
        is_train: bool = True,
        sequence_by_subfolder: bool = False,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.image_size = int(image_size)
        self.is_train = bool(is_train)
        self.sequence_by_subfolder = bool(sequence_by_subfolder)
        self.transform = transform or default_transforms(self.image_size, is_train)

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        self.frames: List[Path] = []
        if self.sequence_by_subfolder:
            for sub in sorted([p for p in self.root.iterdir() if p.is_dir()]):
                self.frames.extend(sorted([p for p in sub.iterdir() if p.is_file() and _is_image(p)]))
        else:
            self.frames = sorted([p for p in self.root.iterdir() if p.is_file() and _is_image(p)])

        if len(self.frames) == 0:
            raise RuntimeError(f"No images found in: {self.root}")

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> torch.Tensor:
        img = Image.open(self.frames[index]).convert("RGB")
        return self.transform(img)


# ------------------------------- VQ-VAE -------------------------------------


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_channels: int = 96, latent_channels: int = 192, downsample_factor: int = 16) -> None:
        super().__init__()
        if downsample_factor not in (4, 8, 16):
            raise ValueError("downsample_factor must be one of {4,8,16}")

        num_down = {4: 2, 8: 3, 16: 4}[downsample_factor]
        layers = [nn.Conv2d(in_channels, hidden_channels, kernel_size=4, stride=2, padding=1)]
        ch = hidden_channels
        for _ in range(num_down - 1):
            layers += [
                nn.ReLU(inplace=True),
                nn.Conv2d(ch, ch * 2, kernel_size=4, stride=2, padding=1),
            ]
            ch *= 2
        layers += [
            ResidualBlock(ch),
            ResidualBlock(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, latent_channels, kernel_size=3, padding=1),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, out_channels: int = 3, hidden_channels: int = 96, latent_channels: int = 192, downsample_factor: int = 16) -> None:
        super().__init__()
        if downsample_factor not in (4, 8, 16):
            raise ValueError("downsample_factor must be one of {4,8,16}")

        num_up = {4: 2, 8: 3, 16: 4}[downsample_factor]
        ch = hidden_channels * (2 ** (num_up - 1))
        layers = [
            nn.Conv2d(latent_channels, ch, kernel_size=3, padding=1),
            ResidualBlock(ch),
            ResidualBlock(ch),
        ]
        for _ in range(num_up - 1):
            layers += [
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(ch, ch // 2, kernel_size=4, stride=2, padding=1),
            ]
            ch //= 2
        layers += [
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ch, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25, use_ema: bool = True, ema_decay: float = 0.99, eps: float = 1e-5, ema_warmup_steps: int = 100, max_codebook_norm: float = 2.0, dead_code_threshold: float = 1.0, dead_code_check_steps: int = 100) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.commitment_cost = float(commitment_cost)
        self.use_ema = bool(use_ema)
        self.ema_decay = float(ema_decay)
        self.eps = float(eps)
        self.ema_warmup_steps = int(ema_warmup_steps)
        self.max_codebook_norm = float(max_codebook_norm)
        self.dead_code_threshold = float(dead_code_threshold)
        self.dead_code_check_steps = int(dead_code_check_steps)
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        # Initialize by embedding_dim scale, not num_embeddings (more stable)
        nn.init.uniform_(self.embedding.weight, -1.0 / self.embedding_dim, 1.0 / self.embedding_dim)
        if self.use_ema:
            self.register_buffer("ema_cluster_size", torch.zeros(self.num_embeddings))
            self.register_buffer("ema_w", self.embedding.weight.data.clone())
        # Track steps to apply EMA warmup
        self.register_buffer("_steps", torch.zeros((), dtype=torch.long))
        # One-time lazy data-dependent init flag
        self.register_buffer("_data_initialized", torch.zeros((), dtype=torch.bool))

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Compute quantization in float32 to avoid NaN/Inf under AMP/FP16
        orig_dtype = z_e.dtype
        b, c, h, w = z_e.shape
        z_e_32 = z_e.float()
        # Stabilize latent scale: standardize per-batch to unit variance
        with torch.no_grad():
            std = z_e_32.std(dim=(0, 2, 3), keepdim=True).clamp_min(self.eps)
        z_e_32 = z_e_32 / std
        z_e_flat = z_e_32.permute(0, 2, 3, 1).contiguous().view(-1, c)
        emb_w = self.embedding.weight.float()

        # Data-dependent codebook init on first forward pass
        if self.training and self.use_ema and (not bool(self._data_initialized.item())):
            with torch.no_grad():
                # Sample without replacement if possible
                num_samples = min(self.num_embeddings, z_e_flat.shape[0])
                idx = torch.randperm(z_e_flat.shape[0], device=z_e_flat.device)[:num_samples]
                self.embedding.weight.data[:num_samples].copy_(z_e_flat[idx])
                if self.use_ema:
                    self.ema_w.data[:num_samples].copy_(z_e_flat[idx])
                    self.ema_cluster_size.data[:num_samples].fill_(1.0)
            self._data_initialized.fill_(True)

        distances = (
            z_e_flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * z_e_flat @ emb_w.t()
            + emb_w.pow(2).sum(dim=1)
        )
        encoding_indices = torch.argmin(distances, dim=1)
        z_q_32 = F.embedding(encoding_indices, emb_w).view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

        commitment = self.commitment_cost * F.mse_loss(z_e_32.detach(), z_q_32)
        # Step counter for warmup
        if self.training:
            self._steps += 1

        use_ema_now = self.use_ema and self.training and (self._steps.item() >= self.ema_warmup_steps)

        if use_ema_now:
            encodings_one_hot = F.one_hot(encoding_indices, self.num_embeddings).to(z_e_flat.dtype)
            cluster_size = encodings_one_hot.sum(dim=0)
            self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size, alpha=1 - self.ema_decay)
            n = self.ema_cluster_size.sum()
            cluster_size = ((self.ema_cluster_size + self.eps) / (n + self.num_embeddings * self.eps)) * n
            embed_sum = encodings_one_hot.t() @ z_e_flat
            self.ema_w.mul_(self.ema_decay).add_(embed_sum, alpha=1 - self.ema_decay)
            updated = self.ema_w / cluster_size.unsqueeze(1)
            # Clamp codebook values to prevent extreme scales
            updated = torch.clamp(updated, min=-self.max_codebook_norm, max=self.max_codebook_norm)
            self.embedding.weight.data.copy_(updated)
            # Dead-code revival
            if self._steps.item() % self.dead_code_check_steps == 0:
                dead_mask = self.ema_cluster_size < self.dead_code_threshold
                if bool(dead_mask.any()):
                    num_dead = int(dead_mask.sum().item())
                    rand_src = z_e_flat[torch.randint(0, z_e_flat.shape[0], (num_dead,), device=z_e_flat.device)]
                    self.embedding.weight.data[dead_mask] = rand_src
                    self.ema_w.data[dead_mask] = rand_src
                    self.ema_cluster_size.data[dead_mask] = self.ema_cluster_size.mean().clamp_min(1.0)
            codebook = torch.tensor(0.0, device=z_e.device, dtype=z_e_32.dtype)
        else:
            # Non-EMA variant: update codebook with gradients
            codebook = F.mse_loss(z_e_32.detach(), z_q_32)
        # Keep VQ loss in float32 for numerical stability; caller may cast if needed
        vq_loss = commitment + codebook

        z_q_st = (z_e_32 + (z_q_32 - z_e_32).detach()).to(orig_dtype)
        encodings_one_hot = F.one_hot(encoding_indices, self.num_embeddings).float()
        avg_probs = encodings_one_hot.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(torch.clamp(avg_probs, min=1e-10))))
        indices = encoding_indices.view(b, h, w)
        return z_q_st, vq_loss, perplexity, indices

    def codes_to_embeddings(self, indices: torch.Tensor) -> torch.Tensor:
        b, h, w = indices.shape
        z_q = self.embedding(indices.view(b, -1)).view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return z_q


@dataclass
class VQVAEConfig:
    image_size: int = 128
    in_channels: int = 3
    hidden_channels: int = 96
    latent_channels: int = 192
    num_embeddings: int = 512
    embedding_dim: int = 128
    downsample_factor: int = 16
    commitment_cost: float = 0.25
    use_ema: bool = True
    ema_decay: float = 0.99
    ema_warmup_steps: int = 100
    max_codebook_norm: float = 2.0
    dead_code_threshold: float = 1.0
    dead_code_check_steps: int = 100


class VQVAE(nn.Module):
    def __init__(self, config: VQVAEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = Encoder(
            in_channels=config.in_channels,
            hidden_channels=config.hidden_channels,
            latent_channels=config.latent_channels,
            downsample_factor=config.downsample_factor,
        )
        self.quantizer = VectorQuantizer(
            num_embeddings=config.num_embeddings,
            embedding_dim=config.embedding_dim,
            commitment_cost=config.commitment_cost,
            use_ema=config.use_ema,
            ema_decay=config.ema_decay,
            ema_warmup_steps=config.ema_warmup_steps,
            max_codebook_norm=config.max_codebook_norm,
            dead_code_threshold=config.dead_code_threshold,
            dead_code_check_steps=config.dead_code_check_steps,
        )
        self.proj_to_code = nn.Conv2d(config.latent_channels, config.embedding_dim, kernel_size=1)
        self.proj_from_code = nn.Conv2d(config.embedding_dim, config.latent_channels, kernel_size=1)
        self.decoder = Decoder(
            out_channels=config.in_channels,
            hidden_channels=config.hidden_channels,
            latent_channels=config.latent_channels,
            downsample_factor=config.downsample_factor,
        )

    def encode_to_indices(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_e = self.encoder(x)
        z_e = self.proj_to_code(z_e)
        _, _, _, indices = self.quantizer(z_e)
        return indices, z_e

    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        z_q = self.quantizer.codes_to_embeddings(indices)
        z = self.proj_from_code(z_q)
        x_rec = self.decoder(z)
        return x_rec

    def forward(self, x: torch.Tensor) -> dict:
        z_e = self.encoder(x)
        z_e = self.proj_to_code(z_e)
        z_q, vq_loss, perplexity, indices = self.quantizer(z_e)
        z = self.proj_from_code(z_q)
        x_rec = self.decoder(z)
        return {"recon": x_rec, "vq_loss": vq_loss, "perplexity": perplexity, "indices": indices}

    def code_grid_size(self, image_size: int) -> int:
        return image_size // self.config.downsample_factor


# ------------------------------- GPT ----------------------------------------


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
    embed_dim: int = 384
    num_layers: int = 6
    num_heads: int = 6
    dropout: float = 0.1
    # Enhancements
    use_pos2d: bool = False
    max_h: int = 0
    max_w: int = 0
    tie_weights: bool = True


class GPTNextFrame(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.embed_dim)
        if cfg.use_pos2d:
            # +1 to reserve 0 for special tokens (BOS/SEP)
            self.row_embed = nn.Embedding(max(1, cfg.max_h + 1), cfg.embed_dim)
            self.col_embed = nn.Embedding(max(1, cfg.max_w + 1), cfg.embed_dim)
        self.seg_embed = nn.Embedding(2, cfg.embed_dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([GPTBlock(cfg.embed_dim, cfg.num_heads, dropout=cfg.dropout) for _ in range(cfg.num_layers)])
        self.ln_f = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.token_embed.weight

    def forward(self, input_ids: torch.Tensor, segment_ids: torch.Tensor, row_ids: Optional[torch.Tensor] = None, col_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t = input_ids.shape
        device = input_ids.device
        if t > self.cfg.max_seq_len:
            raise ValueError(f"Sequence too long: {t} > max_seq_len={self.cfg.max_seq_len}")
        if self.cfg.use_pos2d and (row_ids is not None) and (col_ids is not None):
            x = self.token_embed(input_ids) + self.row_embed(row_ids) + self.col_embed(col_ids) + self.seg_embed(segment_ids)
        else:
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
    def generate(self, prefix_ids: torch.Tensor, prefix_segs: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = 50, top_p: Optional[float] = None, forbidden_token_ids: Optional[Tuple[int, ...]] = None, row_ids_full: Optional[torch.Tensor] = None, col_ids_full: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.eval()
        generated = prefix_ids
        generated_segs = prefix_segs
        device = prefix_ids.device
        for _ in range(max_new_tokens):
            if (row_ids_full is not None) and (col_ids_full is not None):
                cur_len = generated.shape[1]
                logits = self.forward(generated, generated_segs, row_ids=row_ids_full[:, :cur_len], col_ids=col_ids_full[:, :cur_len])
            else:
                logits = self.forward(generated, generated_segs)
            next_token_logits = logits[:, -1, :] / max(1e-6, temperature)
            if forbidden_token_ids is not None:
                next_token_logits[:, list(forbidden_token_ids)] = float("-inf")
            if top_k is not None and top_k > 0:
                topk_vals, topk_idx = torch.topk(next_token_logits, k=min(top_k, next_token_logits.shape[-1]))
                filtered = torch.full_like(next_token_logits, float("-inf"))
                filtered.scatter_(1, topk_idx, topk_vals)
                next_token_logits = filtered
            if top_p is not None and top_p > 0.0 and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_token_logits, descending=True, dim=-1)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                cutoff = (cumprobs > top_p).float().argmax(dim=-1, keepdim=True)
                mask = torch.ones_like(sorted_logits, dtype=torch.bool)
                b_idx = torch.arange(sorted_logits.size(0), device=device).unsqueeze(-1)
                mask[b_idx, cutoff:] = False
                filtered = torch.full_like(next_token_logits, float("-inf"))
                filtered.scatter_(1, sorted_idx[mask].view(next_token_logits.size(0), -1), sorted_logits[mask].view(next_token_logits.size(0), -1))
                next_token_logits = filtered
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_seg = torch.ones_like(next_token)
            generated = torch.cat([generated, next_token], dim=1)
            generated_segs = torch.cat([generated_segs, next_seg], dim=1)
        return generated


# --------------------------- Training Utilities -----------------------------


def save_image_grid(x_in: torch.Tensor, x_rec: torch.Tensor, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    x_in = torch.nan_to_num(torch.clamp((x_in + 1.0) * 0.5, 0.0, 1.0))
    x_rec = torch.nan_to_num(torch.clamp((x_rec + 1.0) * 0.5, 0.0, 1.0))
    grid = torch.cat([x_in, x_rec], dim=3)[0]
    T.ToPILImage()(grid.cpu()).save(out_dir / name)


def _to01(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(torch.clamp((x + 1.0) * 0.5, 0.0, 1.0))


def ssim_value(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # convert to [-1,1] expected by our ssim_loss_simple
    return 1.0 - ssim_loss_simple(x, y)


def psnr_value(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # compute on [0,1]
    x01 = _to01(x)
    y01 = _to01(y)
    mse = F.mse_loss(x01, y01)
    return 10.0 * torch.log10(torch.tensor(1.0, device=x.device) / (mse + 1e-8))


# ------------------------------ Phase 1: AE ---------------------------------


@dataclass
class AEConfig:
    image_size: int = 128
    in_channels: int = 3
    base_channels: int = 64
    bottleneck_channels: int = 512


class ConvAutoencoder(nn.Module):
    def __init__(self, cfg: AEConfig) -> None:
        super().__init__()
        c = cfg.base_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(cfg.in_channels, c, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c * 2, 4, 2, 1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * 2, c * 4, 4, 2, 1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * 4, cfg.bottleneck_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(cfg.bottleneck_channels, c * 4, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c * 4, c * 2, 4, 2, 1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c * 2, c, 4, 2, 1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c, cfg.in_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


def ssim_loss_simple(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_x = x.mean(dim=[2, 3], keepdim=True)
    mu_y = y.mean(dim=[2, 3], keepdim=True)
    sigma_x = x.var(dim=[2, 3], keepdim=True)
    sigma_y = y.var(dim=[2, 3], keepdim=True)
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean(dim=[2, 3], keepdim=True)
    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    return (1 - ssim).mean()


class UNetAutoencoder(nn.Module):
    def __init__(self, cfg: AEConfig) -> None:
        super().__init__()
        c = cfg.base_channels
        # Encoder
        self.e1 = nn.Sequential(nn.Conv2d(cfg.in_channels, c, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(c, c, 3, 1, 1), nn.ReLU(inplace=True))
        self.d1 = nn.Conv2d(c, c, 4, 2, 1)
        self.e2 = nn.Sequential(nn.Conv2d(c, c * 2, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(c * 2, c * 2, 3, 1, 1), nn.ReLU(inplace=True))
        self.d2 = nn.Conv2d(c * 2, c * 2, 4, 2, 1)
        self.e3 = nn.Sequential(nn.Conv2d(c * 2, c * 4, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(c * 4, c * 4, 3, 1, 1), nn.ReLU(inplace=True))
        self.bottleneck = nn.Sequential(nn.Conv2d(c * 4, cfg.bottleneck_channels, 3, 1, 1), nn.ReLU(inplace=True))

        # Decoder
        self.u3 = nn.ConvTranspose2d(cfg.bottleneck_channels, c * 4, 4, 2, 1)  # 64->128
        self.d3 = nn.Sequential(nn.Conv2d(c * 6, c * 4, 3, 1, 1), nn.ReLU(inplace=True))  # concat with s2 (2c)
        self.u2 = nn.ConvTranspose2d(c * 4, c * 2, 4, 2, 1)  # 128->256
        self.d4 = nn.Sequential(nn.Conv2d(c * 3, c * 2, 3, 1, 1), nn.ReLU(inplace=True))  # concat with s1 (c)
        self.out = nn.Sequential(nn.Conv2d(c * 2, c, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(c, cfg.in_channels, 3, 1, 1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.e1(x)  # 256, c
        p1 = F.relu(self.d1(s1))  # 128, c
        s2 = self.e2(p1)  # 128, 2c
        p2 = F.relu(self.d2(s2))  # 64, 2c
        s3 = self.e3(p2)  # 64, 4c
        b = self.bottleneck(s3)  # 64, bottleneck
        x = self.u3(b)  # 128, 4c
        x = self.d3(torch.cat([x, s2], dim=1))  # 128, 4c
        x = self.u2(x)  # 256, 2c
        x = self.d4(torch.cat([x, s1], dim=1))  # 256, 2c
        x = self.out(x)  # 256, 3 -> tanh
        return x


@dataclass
class EditorUNetConfig:
    image_size: int = 128
    base_channels: int = 64
    bottleneck_channels: int = 512
    num_instructions: int = 8


class UNetEditor(nn.Module):
    def __init__(self, cfg: EditorUNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        in_ch = 3 + 1 + cfg.num_instructions  # image + mask + instruction one-hot planes
        c = cfg.base_channels
        # Encoder
        self.e1 = nn.Sequential(nn.Conv2d(in_ch, c, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(c, c, 3, 1, 1), nn.ReLU(inplace=True))
        self.d1 = nn.Conv2d(c, c, 4, 2, 1)
        self.e2 = nn.Sequential(nn.Conv2d(c, c * 2, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(c * 2, c * 2, 3, 1, 1), nn.ReLU(inplace=True))
        self.d2 = nn.Conv2d(c * 2, c * 2, 4, 2, 1)
        self.e3 = nn.Sequential(nn.Conv2d(c * 2, c * 4, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(c * 4, c * 4, 3, 1, 1), nn.ReLU(inplace=True))
        self.bottleneck = nn.Sequential(nn.Conv2d(c * 4, cfg.bottleneck_channels, 3, 1, 1), nn.ReLU(inplace=True))

        # Decoder
        self.u3 = nn.ConvTranspose2d(cfg.bottleneck_channels, c * 4, 4, 2, 1)
        self.d3 = nn.Sequential(nn.Conv2d(c * 6, c * 4, 3, 1, 1), nn.ReLU(inplace=True))
        self.u2 = nn.ConvTranspose2d(c * 4, c * 2, 4, 2, 1)
        self.d4 = nn.Sequential(nn.Conv2d(c * 3, c * 2, 3, 1, 1), nn.ReLU(inplace=True))
        self.out = nn.Sequential(nn.Conv2d(c * 2, c, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(c, 3, 3, 1, 1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.e1(x)
        p1 = F.relu(self.d1(s1))
        s2 = self.e2(p1)
        p2 = F.relu(self.d2(s2))
        s3 = self.e3(p2)
        b = self.bottleneck(s3)
        x = self.u3(b)
        x = self.d3(torch.cat([x, s2], dim=1))
        x = self.u2(x)
        x = self.d4(torch.cat([x, s1], dim=1))
        x = self.out(x)
        return x

    def init_from_unet_autoencoder(self, ae_state: dict) -> None:
        # Copy weights where shapes match exactly
        own = self.state_dict()
        for k, v in ae_state.items():
            if k.startswith("e1.0.weight"):
                # special case: first conv has extra channels
                w = own["e1.0.weight"]
                with torch.no_grad():
                    w.zero_()
                    src = v
                    copy_ch = min(src.shape[1], 3)
                    w[:, :copy_ch] = src[:, :copy_ch]
                continue
            if k in own and own[k].shape == v.shape:
                own[k].copy_(v)
        self.load_state_dict(own)


class UNetTranslator(nn.Module):
    """Image-to-image UNet that maps x_t -> x_{t+1}. Matches UNetAutoencoder toplogy."""
    def __init__(self, image_size: int = 128, base_channels: int = 64, bottleneck_channels: int = 512) -> None:
        super().__init__()
        c = base_channels
        self.in_block = nn.Sequential(
            nn.Conv2d(6, c, 3, 1, 1),  # concatenate x_t and (optional) warped/aux; here x_t + residual placeholder
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.e1 = nn.Identity()
        self.d1 = nn.Conv2d(c, c, 4, 2, 1)
        self.e2 = nn.Sequential(nn.Conv2d(c, c * 2, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(c * 2, c * 2, 3, 1, 1), nn.ReLU(inplace=True))
        self.d2 = nn.Conv2d(c * 2, c * 2, 4, 2, 1)
        self.e3 = nn.Sequential(nn.Conv2d(c * 2, c * 4, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(c * 4, c * 4, 3, 1, 1), nn.ReLU(inplace=True))
        self.bottleneck = nn.Sequential(nn.Conv2d(c * 4, bottleneck_channels, 3, 1, 1), nn.ReLU(inplace=True))

        self.u3 = nn.ConvTranspose2d(bottleneck_channels, c * 4, 4, 2, 1)
        self.d3 = nn.Sequential(nn.Conv2d(c * 6, c * 4, 3, 1, 1), nn.ReLU(inplace=True))
        self.u2 = nn.ConvTranspose2d(c * 4, c * 2, 4, 2, 1)
        self.d4 = nn.Sequential(nn.Conv2d(c * 3, c * 2, 3, 1, 1), nn.ReLU(inplace=True))
        self.out = nn.Sequential(nn.Conv2d(c * 2, c, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(c, 3, 3, 1, 1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.in_block(x)
        p1 = F.relu(self.d1(s1))
        s2 = self.e2(p1)
        p2 = F.relu(self.d2(s2))
        s3 = self.e3(p2)
        b = self.bottleneck(s3)
        x = self.u3(b)
        x = self.d3(torch.cat([x, s2], dim=1))
        x = self.u2(x)
        x = self.d4(torch.cat([x, s1], dim=1))
        x = self.out(x)
        return x

    def init_from_unet_autoencoder(self, ae_state: dict) -> None:
        own = self.state_dict()
        for k, v in ae_state.items():
            if k in own and own[k].shape == v.shape:
                own[k].copy_(v)
        self.load_state_dict(own)


def sobel_edge_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # x,y in [-1,1]; compute gradients per-channel
    kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    kx = kx.repeat(x.shape[1], 1, 1, 1)
    ky = ky.repeat(x.shape[1], 1, 1, 1)
    gx_x = F.conv2d(x, kx, padding=1, groups=x.shape[1])
    gy_x = F.conv2d(x, ky, padding=1, groups=x.shape[1])
    gx_y = F.conv2d(y, kx, padding=1, groups=y.shape[1])
    gy_y = F.conv2d(y, ky, padding=1, groups=y.shape[1])
    mag_x = torch.sqrt(gx_x ** 2 + gy_x ** 2 + 1e-6)
    mag_y = torch.sqrt(gx_y ** 2 + gy_y ** 2 + 1e-6)
    return F.l1_loss(mag_x, mag_y)


def make_random_rect_mask(h: int, w: int) -> torch.Tensor:
    """Return a 1xHxW binary mask with 1s in a random rectangle (possibly two)."""
    mask = torch.zeros(1, h, w, dtype=torch.float32)
    for _ in range(random.randint(1, 2)):
        rh = max(8, int(random.uniform(0.2, 0.6) * h))
        rw = max(8, int(random.uniform(0.2, 0.6) * w))
        top = random.randint(0, max(0, h - rh))
        left = random.randint(0, max(0, w - rw))
        mask[:, top : top + rh, left : left + rw] = 1.0
    return mask


def make_random_edit_mask(h: int, w: int, full_prob: float = 0.2) -> torch.Tensor:
    """Return a 1xHxW binary mask; with probability full_prob it covers the full image."""
    if random.random() < full_prob:
        return torch.ones(1, h, w, dtype=torch.float32)
    return make_random_rect_mask(h, w)


def apply_instruction_masked(x01: torch.Tensor, mask01: torch.Tensor, instr_id: int) -> torch.Tensor:
    """Apply an edit specified by instr_id inside mask on an image tensor in [0,1].

    x01: (B,C,H,W) in [0,1], mask01: (B,1,H,W) in {0,1}
    """
    b = x01.shape[0]
    factor_a = random.uniform(1.1, 1.4)
    factor_b = random.uniform(0.6, 0.9)
    if instr_id == 0:  # brighten
        edited = torch.clamp(x01 * factor_a, 0.0, 1.0)
    elif instr_id == 1:  # darken
        edited = torch.clamp(x01 * factor_b, 0.0, 1.0)
    elif instr_id == 2:  # saturate up
        edited = TF.adjust_saturation(x01, saturation_factor=random.uniform(1.2, 1.6))
    elif instr_id == 3:  # desaturate
        edited = TF.adjust_saturation(x01, saturation_factor=random.uniform(0.5, 0.8))
    elif instr_id == 4:  # hue shift +
        edited = TF.adjust_hue(x01, hue_factor=random.uniform(0.02, 0.12))
    elif instr_id == 5:  # hue shift -
        edited = TF.adjust_hue(x01, hue_factor=-random.uniform(0.02, 0.12))
    elif instr_id == 6:  # contrast up
        edited = TF.adjust_contrast(x01, contrast_factor=random.uniform(1.2, 1.6))
    elif instr_id == 7:  # contrast down
        edited = TF.adjust_contrast(x01, contrast_factor=random.uniform(0.5, 0.8))
    else:
        edited = x01
    return x01 * (1.0 - mask01) + edited * mask01


INSTRUCTION_NAMES = (
    "brighten",
    "darken",
    "saturate",
    "desaturate",
    "hue_pos",
    "hue_neg",
    "contrast_up",
    "contrast_down",
)


def instruction_name_to_id(name: str) -> int:
    low = name.strip().lower()
    if low not in INSTRUCTION_NAMES:
        raise ValueError(f"Unknown instruction '{name}'. Allowed: {', '.join(INSTRUCTION_NAMES)}")
    return INSTRUCTION_NAMES.index(low)


def build_editor_condition(mask01: torch.Tensor, instr_ids: torch.Tensor, image_h: int, image_w: int, num_instr: int) -> torch.Tensor:
    """Create per-pixel condition planes: 1 mask + one-hot instruction planes.

    Returns (B, 1 + num_instr, H, W) in {0,1}.
    """
    b = mask01.shape[0]
    cond = torch.zeros(b, 1 + num_instr, image_h, image_w, device=mask01.device, dtype=mask01.dtype)
    cond[:, 0:1] = mask01
    for i in range(b):
        instr = int(instr_ids[i].item())
        cond[i, 1 + instr : 1 + instr + 1] = 1.0
    return cond

def train_autoencoder_phase1(
    data_root: str,
    out_path: str,
    image_size: int = 128,
    batch_size: int = 16,
    epochs: int = 20,
    lr: float = 1e-4,
    ssim_weight: float = 0.5,
    base_channels: int = 64,
    bottleneck_channels: int = 512,
    use_unet: bool = False,
    edge_weight: float = 0.0,
    device: str = "cuda",
) -> str:
    device_t = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    ds = FramePairDataset(root=data_root, image_size=image_size, is_train=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    cfg = AEConfig(image_size=image_size, in_channels=3, base_channels=base_channels, bottleneck_channels=bottleneck_channels)
    model = (UNetAutoencoder(cfg) if use_unet else ConvAutoencoder(cfg)).to(device_t)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    l1 = nn.L1Loss()

    model.train()
    last_batch_x = None
    for epoch in range(1, epochs + 1):
        pbar = tqdm(dl, desc=f"AE {epoch}/{epochs}")
        for x_t, _ in pbar:
            x = x_t.to(device_t, non_blocking=True)
            last_batch_x = x
            recon = model(x)
            loss_l1 = l1(recon, x)
            loss_ssim = ssim_loss_simple(recon, x)
            loss_edge = sobel_edge_loss(recon, x) if edge_weight > 0.0 else torch.tensor(0.0, device=device_t)
            loss = loss_l1 + ssim_weight * loss_ssim + edge_weight * loss_edge
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            pbar.set_postfix({"l1": float(loss_l1), "ssim": float(loss_ssim), "edge": float(loss_edge)})

        if last_batch_x is None:
            continue
        model.eval()
        with torch.no_grad():
            out_vis = model(last_batch_x[:4])
        save_image_grid(last_batch_x[:4], out_vis, Path("samples/ae"), f"epoch_{epoch:03d}.png")
        model.train()

    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)
    # Save architecture type to allow correct loading later
    arch = "unet" if use_unet else "conv"
    torch.save({"state_dict": model.state_dict(), "config": cfg.__dict__, "arch": arch}, out_path_p)
    return str(out_path_p)


@torch.no_grad()
def reconstruct_image_phase1(image_path: str, ckpt_path: str, out_path: str, device: str = "cuda") -> str:
    device_t = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = AEConfig(**ckpt["config"])  # type: ignore[arg-type]
    # Determine architecture: prefer explicit field, otherwise infer from keys
    state_dict = ckpt["state_dict"]  # type: ignore[index]
    arch = ckpt.get("arch", None)
    if arch is None:
        keys = list(state_dict.keys())
        uses_unet = any(k.startswith("e1.") or k.startswith("bottleneck.") or k.startswith("out.") for k in keys)
        arch = "unet" if uses_unet else "conv"
    if arch == "unet":
        model = UNetAutoencoder(cfg).to(device_t)
    else:
        model = ConvAutoencoder(cfg).to(device_t)
    model.load_state_dict(state_dict)  # type: ignore[arg-type]
    model.eval()

    tfm = T.Compose([
        T.Resize(cfg.image_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop(cfg.image_size),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device_t)
    recon = model(x)
    out_dir = Path(out_path)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    save_image_grid(x, recon, out_dir.parent, Path(out_path).name)
    return out_path

def train_vqvae(
    data_root: str,
    out_path: str,
    image_size: int = 128,
    batch_size: int = 8,
    epochs: int = 8,
    lr: float = 3e-4,
    downsample_factor: int = 16,
    num_embeddings: int = 512,
    embedding_dim: int = 128,
    hidden_channels: int = 96,
    latent_channels: int = 192,
    commitment_cost: float = 0.25,
    ema_decay: float = 0.99,
    ema_warmup_steps: int = 100,
    max_codebook_norm: float = 2.0,
    dead_code_threshold: float = 1.0,
    dead_code_check_steps: int = 100,
    recon_weight: float = 1.0,
    device: str = "cuda",
) -> str:
    device_t = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    ds = FramePairDataset(root=data_root, image_size=image_size, is_train=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    cfg = VQVAEConfig(
        image_size=image_size,
        in_channels=3,
        hidden_channels=hidden_channels,
        latent_channels=latent_channels,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        downsample_factor=downsample_factor,
        commitment_cost=commitment_cost,
        use_ema=True,
        ema_decay=ema_decay,
        ema_warmup_steps=ema_warmup_steps,
        max_codebook_norm=max_codebook_norm,
        dead_code_threshold=dead_code_threshold,
        dead_code_check_steps=dead_code_check_steps,
    )
    model = VQVAE(cfg).to(device_t)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    recon_crit = nn.L1Loss()

    model.train()
    last_batch_x = None
    for epoch in range(1, epochs + 1):
        pbar = tqdm(dl, desc=f"VQ-VAE {epoch}/{epochs}")
        for x_t, _ in pbar:
            x = x_t.to(device_t, non_blocking=True)
            last_batch_x = x
            # Use full precision for stability in VQ-VAE training
            with torch.amp.autocast(device_type=device_t.type, enabled=False):
                out = model(x)
                recon = out["recon"]
                vq_loss = out["vq_loss"]
                recon_loss = recon_crit(recon, x)
                loss = recon_weight * recon_loss + vq_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            pbar.set_postfix({"recon": float(recon_loss), "vq": float(vq_loss), "perp": float(out["perplexity"])})

        # Save sample at epoch end
        if last_batch_x is None:
            continue
        model.eval()
        with torch.no_grad():
            out_vis = model(last_batch_x[:4])
        save_image_grid(last_batch_x[:4], out_vis["recon"], Path("samples/vqvae"), f"epoch_{epoch:03d}.png")
        model.train()

    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": model.config.__dict__}, out_path_p)
    return str(out_path_p)


def train_gpt_next(data_root: str, vqvae_ckpt: str, out_path: str, image_size: int = 128, batch_size: int = 8, epochs: int = 8, lr: float = 3e-4, embed_dim: int = 384, layers: int = 6, heads: int = 6, device: str = "cuda") -> Tuple[str, dict]:
    device_t = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    # Load VQ-VAE
    ckpt = torch.load(vqvae_ckpt, map_location="cpu")
    vqcfg = VQVAEConfig(**ckpt["config"])  # type: ignore[arg-type]
    vqvae = VQVAE(vqcfg).to(device_t)
    vqvae.load_state_dict(ckpt["state_dict"])  # type: ignore[index]
    vqvae.eval()

    ds = FramePairDataset(root=data_root, image_size=image_size, is_train=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    sample_x, _ = next(iter(dl))
    with torch.no_grad():
        sample_indices, _ = vqvae.encode_to_indices(sample_x.to(device_t)[:1])
    h_code, w_code = sample_indices.shape[-2], sample_indices.shape[-1]
    tokens_per_frame = h_code * w_code

    num_embeddings = vqcfg.num_embeddings
    bos_id = num_embeddings
    sep_id = num_embeddings + 1
    vocab_size = num_embeddings + 2
    max_seq_len = 1 + tokens_per_frame + 1 + tokens_per_frame
    gptcfg = GPTConfig(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        embed_dim=embed_dim,
        num_layers=layers,
        num_heads=heads,
        dropout=0.1,
        use_pos2d=True,
        max_h=h_code,
        max_w=w_code,
        tie_weights=True,
    )
    gpt = GPTNextFrame(gptcfg).to(device_t)

    scaler = torch.amp.GradScaler(device_t.type)
    opt = torch.optim.AdamW(gpt.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    # Warmup + cosine scheduler
    total_steps = max(1, epochs * (len(dl)))
    warmup_steps = max(1, int(0.1 * total_steps))
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    last_batch = None
    for epoch in range(1, epochs + 1):
        gpt.train()
        pbar = tqdm(dl, desc=f"GPT {epoch}/{epochs}")
        for x_t, x_tp1 in pbar:
            x_t = x_t.to(device_t, non_blocking=True)
            x_tp1 = x_tp1.to(device_t, non_blocking=True)
            last_batch = (x_t, x_tp1)
            with torch.no_grad():
                ids_t, _ = vqvae.encode_to_indices(x_t)
                ids_tp1, _ = vqvae.encode_to_indices(x_tp1)
            b = ids_t.shape[0]
            prev_tokens = ids_t.view(b, -1)
            next_tokens = ids_tp1.view(b, -1)

            bos = torch.full((b, 1), bos_id, dtype=torch.long, device=device_t)
            sep = torch.full((b, 1), sep_id, dtype=torch.long, device=device_t)
            input_ids = torch.cat([bos, prev_tokens, sep, next_tokens[:, :-1]], dim=1)
            seg_prev = torch.zeros((b, 1 + tokens_per_frame), dtype=torch.long, device=device_t)
            seg_next = torch.ones((b, 1 + tokens_per_frame - 1), dtype=torch.long, device=device_t)
            segment_ids = torch.cat([seg_prev, seg_next], dim=1)

            # Build 2D positions for the whole sequence (BOS + prev + SEP + tgt[:-1])
            # We set row/col=0 for BOS/SEP, and (1..h_code)/(1..w_code) for tokens
            rows = torch.zeros_like(input_ids)
            cols = torch.zeros_like(input_ids)
            start = 1  # skip BOS
            # prev_tokens positions
            rc = torch.arange(tokens_per_frame, device=device_t)
            r = (rc // w_code) + 1
            c = (rc % w_code) + 1
            rows[:, start:start+tokens_per_frame] = r.view(1, -1).expand(b, -1)
            cols[:, start:start+tokens_per_frame] = c.view(1, -1).expand(b, -1)
            start += tokens_per_frame
            # SEP stays 0
            start += 1
            # target[:-1] positions
            rc2 = torch.arange(tokens_per_frame - 1, device=device_t)
            r2 = (rc2 // w_code) + 1
            c2 = (rc2 % w_code) + 1
            rows[:, start:start+tokens_per_frame-1] = r2.view(1, -1).expand(b, -1)
            cols[:, start:start+tokens_per_frame-1] = c2.view(1, -1).expand(b, -1)

            with torch.amp.autocast(device_type=device_t.type, enabled=True):
                logits = gpt(input_ids=input_ids, segment_ids=segment_ids, row_ids=rows, col_ids=cols)
                targets = torch.full_like(input_ids, fill_value=-100)
                targets[:, -tokens_per_frame:] = next_tokens
                loss = ce_loss(logits.view(-1, logits.size(-1)), targets.view(-1))

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            pbar.set_postfix({"loss": float(loss)})

        # Qualitative sample
        if last_batch is not None:
            gpt.eval()
            x_vis_t = last_batch[0][:1]
            with torch.no_grad():
                ids_t, _ = vqvae.encode_to_indices(x_vis_t)
                prev_tokens = ids_t.view(1, -1)
                bos = torch.full((1, 1), bos_id, dtype=torch.long, device=device_t)
                sep = torch.full((1, 1), sep_id, dtype=torch.long, device=device_t)
                prefix_ids = torch.cat([bos, prev_tokens, sep], dim=1)
                seg_prev = torch.zeros((1, 1 + prev_tokens.shape[1]), dtype=torch.long, device=device_t)
                seg_sep = torch.ones((1, 1), dtype=torch.long, device=device_t)
                prefix_segs = torch.cat([seg_prev, seg_sep], dim=1)
                # Prepare full row/col ids for prefix + generation
                total_len = int(prefix_ids.shape[1] + tokens_per_frame)
                rows_full = torch.zeros((1, total_len), dtype=torch.long, device=device_t)
                cols_full = torch.zeros((1, total_len), dtype=torch.long, device=device_t)
                # fill prev positions
                rc = torch.arange(tokens_per_frame, device=device_t)
                r = (rc // w_code) + 1
                c = (rc % w_code) + 1
                rows_full[:, 1:1+tokens_per_frame] = r.view(1, -1)
                cols_full[:, 1:1+tokens_per_frame] = c.view(1, -1)
                # future tokens positions
                rc2 = torch.arange(tokens_per_frame, device=device_t)
                r2 = (rc2 // w_code) + 1
                c2 = (rc2 % w_code) + 1
                rows_full[:, prefix_ids.shape[1]:] = r2.view(1, -1)
                cols_full[:, prefix_ids.shape[1]:] = c2.view(1, -1)
                generated = gpt.generate(prefix_ids, prefix_segs, max_new_tokens=tokens_per_frame, temperature=0.9, top_k=50, top_p=None, forbidden_token_ids=(bos_id, sep_id), row_ids_full=rows_full, col_ids_full=cols_full)
                next_tokens = generated[:, -tokens_per_frame:]
                tokens_reshaped = next_tokens.view(1, h_code, w_code)
                x_rec = vqvae.decode_from_indices(tokens_reshaped)
            save_image_grid(x_vis_t, x_rec, Path("samples/next_frame"), f"epoch_{epoch:03d}.png")

    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "tokens_per_frame": tokens_per_frame,
        "h_code": h_code,
        "w_code": w_code,
        "bos_id": bos_id,
        "sep_id": sep_id,
        "num_embeddings": num_embeddings,
        "image_size": image_size,
        "use_pos2d": True,
    }
    torch.save({"state_dict": gpt.state_dict(), "config": gpt.cfg.__dict__, "meta": meta}, out_path_p)
    return str(out_path_p), meta


@torch.no_grad()
def predict_autoregressive(image_path: str, vqvae_ckpt: str, gpt_ckpt: str, steps: int = 6, temperature: float = 0.9, top_k: int = 50, out_dir: str = "outputs_next", device: str = "cuda") -> None:
    device_t = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    vq_ckpt = torch.load(vqvae_ckpt, map_location="cpu")
    vqcfg = VQVAEConfig(**vq_ckpt["config"])  # type: ignore[arg-type]
    vqvae = VQVAE(vqcfg).to(device_t)
    vqvae.load_state_dict(vq_ckpt["state_dict"])  # type: ignore[index]
    vqvae.eval()

    gpt_ckpt = torch.load(gpt_ckpt, map_location="cpu")
    gptcfg = GPTConfig(**gpt_ckpt["config"])  # type: ignore[arg-type]
    gpt = GPTNextFrame(gptcfg).to(device_t)
    gpt.load_state_dict(gpt_ckpt["state_dict"])  # type: ignore[index]
    gpt.eval()

    meta = gpt_ckpt["meta"]  # type: ignore[index]
    tokens_per_frame = int(meta["tokens_per_frame"])  # type: ignore[index]
    h_code = int(meta["h_code"])  # type: ignore[index]
    w_code = int(meta["w_code"])  # type: ignore[index]
    bos_id = int(meta["bos_id"])  # type: ignore[index]
    sep_id = int(meta["sep_id"])  # type: ignore[index]
    num_embeddings = int(meta["num_embeddings"])  # type: ignore[index]
    image_size = int(meta["image_size"])  # type: ignore[index]
    # Optional 2D pos info (backward compatibility)
    max_h = int(meta.get("h_code", h_code))
    max_w = int(meta.get("w_code", w_code))

    img = Image.open(image_path).convert("RGB")
    tfm = T.Compose([T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True), T.CenterCrop(image_size), T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    x0 = tfm(img).unsqueeze(0).to(device_t)
    prev_ids, _ = vqvae.encode_to_indices(x0)
    b = prev_ids.shape[0]
    prev_tokens = prev_ids.view(b, -1)
    outputs: List[Image.Image] = []
    current_prev = prev_tokens
    for _ in range(steps):
        bos = torch.full((b, 1), bos_id, dtype=torch.long, device=device_t)
        sep = torch.full((b, 1), sep_id, dtype=torch.long, device=device_t)
        prefix_ids = torch.cat([bos, current_prev, sep], dim=1)
        seg_prev = torch.zeros((b, 1 + current_prev.shape[1]), dtype=torch.long, device=device_t)
        seg_sep = torch.ones((b, 1), dtype=torch.long, device=device_t)
        prefix_segs = torch.cat([seg_prev, seg_sep], dim=1)
        # Build 2D pos if model expects it
        use_pos2d = getattr(gpt.cfg, "use_pos2d", False)
        if use_pos2d:
            total_len = int(prefix_ids.shape[1] + tokens_per_frame)
            rows_full = torch.zeros((b, total_len), dtype=torch.long, device=device_t)
            cols_full = torch.zeros((b, total_len), dtype=torch.long, device=device_t)
            rc = torch.arange(current_prev.shape[1], device=device_t)
            r = (rc // max_w) + 1
            c = (rc % max_w) + 1
            rows_full[:, 1:1+current_prev.shape[1]] = r.view(1, -1).expand(b, -1)
            cols_full[:, 1:1+current_prev.shape[1]] = c.view(1, -1).expand(b, -1)
            rc2 = torch.arange(tokens_per_frame, device=device_t)
            r2 = (rc2 // max_w) + 1
            c2 = (rc2 % max_w) + 1
            rows_full[:, prefix_ids.shape[1]:] = r2.view(1, -1).expand(b, -1)
            cols_full[:, prefix_ids.shape[1]:] = c2.view(1, -1).expand(b, -1)
            generated = gpt.generate(prefix_ids, prefix_segs, max_new_tokens=tokens_per_frame, temperature=temperature, top_k=top_k, top_p=None, forbidden_token_ids=(bos_id, sep_id), row_ids_full=rows_full, col_ids_full=cols_full)
        else:
            generated = gpt.generate(prefix_ids, prefix_segs, max_new_tokens=tokens_per_frame, temperature=temperature, top_k=top_k, forbidden_token_ids=(bos_id, sep_id))
        next_tokens = generated[:, -tokens_per_frame:]
        tokens_reshaped = next_tokens.view(1, h_code, w_code)
        x_rec = vqvae.decode_from_indices(tokens_reshaped)
        img_next = T.ToPILImage()(torch.clamp((x_rec + 1.0) * 0.5, 0.0, 1.0)[0].cpu())
        outputs.append(img_next)
        current_prev = next_tokens

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(outputs):
        im.save(out_dir_p / f"next_{i:03d}.png")


def evaluate_next_frame(
    data_root: str,
    vqvae_ckpt: str,
    gpt_ckpt: str,
    num_batches: int = 10,
    image_size: int = 128,
    batch_size: int = 4,
    temperature: float = 0.9,
    top_k: int = 50,
    device: str = "cuda",
) -> dict:
    """Compute average PSNR/SSIM between predicted next frames and ground truth frames."""
    device_t = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    vq_ckpt = torch.load(vqvae_ckpt, map_location="cpu")
    vqcfg = VQVAEConfig(**vq_ckpt["config"])  # type: ignore[arg-type]
    vqvae = VQVAE(vqcfg).to(device_t)
    vqvae.load_state_dict(vq_ckpt["state_dict"])  # type: ignore[index]
    vqvae.eval()

    gpt_ckpt_d = torch.load(gpt_ckpt, map_location="cpu")
    gptcfg = GPTConfig(**gpt_ckpt_d["config"])  # type: ignore[arg-type]
    gpt = GPTNextFrame(gptcfg).to(device_t)
    gpt.load_state_dict(gpt_ckpt_d["state_dict"])  # type: ignore[index]
    gpt.eval()

    meta = gpt_ckpt_d["meta"]  # type: ignore[index]
    tokens_per_frame = int(meta["tokens_per_frame"])  # type: ignore[index]
    h_code = int(meta["h_code"])  # type: ignore[index]
    w_code = int(meta["w_code"])  # type: ignore[index]
    bos_id = int(meta["bos_id"])  # type: ignore[index]
    sep_id = int(meta["sep_id"])  # type: ignore[index]
    image_size = int(meta["image_size"])  # type: ignore[index]
    use_pos2d = getattr(gpt.cfg, "use_pos2d", False)

    ds = FramePairDataset(root=data_root, image_size=image_size, is_train=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    psnr_list: List[float] = []
    ssim_list: List[float] = []
    processed = 0

    for x_t, x_tp1 in dl:
        x_t = x_t.to(device_t)
        x_tp1 = x_tp1.to(device_t)
        with torch.no_grad():
            ids_t, _ = vqvae.encode_to_indices(x_t)
            prev_tokens = ids_t.view(x_t.shape[0], -1)
            bos = torch.full((x_t.shape[0], 1), bos_id, dtype=torch.long, device=device_t)
            sep = torch.full((x_t.shape[0], 1), sep_id, dtype=torch.long, device=device_t)
            prefix_ids = torch.cat([bos, prev_tokens, sep], dim=1)
            seg_prev = torch.zeros((x_t.shape[0], 1 + prev_tokens.shape[1]), dtype=torch.long, device=device_t)
            seg_sep = torch.ones((x_t.shape[0], 1), dtype=torch.long, device=device_t)
            prefix_segs = torch.cat([seg_prev, seg_sep], dim=1)
            if use_pos2d:
                total_len = int(prefix_ids.shape[1] + tokens_per_frame)
                rows_full = torch.zeros((x_t.shape[0], total_len), dtype=torch.long, device=device_t)
                cols_full = torch.zeros((x_t.shape[0], total_len), dtype=torch.long, device=device_t)
                rc = torch.arange(prev_tokens.shape[1], device=device_t)
                r = (rc // w_code) + 1
                c = (rc % w_code) + 1
                rows_full[:, 1:1+prev_tokens.shape[1]] = r.view(1, -1).expand(x_t.shape[0], -1)
                cols_full[:, 1:1+prev_tokens.shape[1]] = c.view(1, -1).expand(x_t.shape[0], -1)
                rc2 = torch.arange(tokens_per_frame, device=device_t)
                r2 = (rc2 // w_code) + 1
                c2 = (rc2 % w_code) + 1
                rows_full[:, prefix_ids.shape[1]:] = r2.view(1, -1).expand(x_t.shape[0], -1)
                cols_full[:, prefix_ids.shape[1]:] = c2.view(1, -1).expand(x_t.shape[0], -1)
                generated = gpt.generate(prefix_ids, prefix_segs, max_new_tokens=tokens_per_frame, temperature=temperature, top_k=top_k, top_p=None, forbidden_token_ids=(bos_id, sep_id), row_ids_full=rows_full, col_ids_full=cols_full)
            else:
                generated = gpt.generate(prefix_ids, prefix_segs, max_new_tokens=tokens_per_frame, temperature=temperature, top_k=top_k, forbidden_token_ids=(bos_id, sep_id))
            next_tokens = generated[:, -tokens_per_frame:]
            tokens_reshaped = next_tokens.view(x_t.shape[0], h_code, w_code)
            x_pred = vqvae.decode_from_indices(tokens_reshaped)

        for i in range(x_t.shape[0]):
            psnr_list.append(float(psnr_value(x_pred[i:i+1], x_tp1[i:i+1]).item()))
            ssim_list.append(float(ssim_value(x_pred[i:i+1], x_tp1[i:i+1]).item()))

        processed += 1
        if processed >= num_batches:
            break

    return {"psnr": sum(psnr_list)/max(1,len(psnr_list)), "ssim": sum(ssim_list)/max(1,len(ssim_list))}


def train_predictor_ae(
    data_root: str,
    ae_ckpt: str,
    out_path: str,
    image_size: int = 128,
    batch_size: int = 8,
    epochs: int = 8,
    lr: float = 1e-4,
    base_channels: int = 64,
    bottleneck_channels: int = 512,
    device: str = "cuda",
) -> str:
    # Note: ae_ckpt is ignored by design in this refactor (no AE usage)
    device_t = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    # Load exactly two frames: frame_0001.png (input) and frame_0002.png (target)
    data_root_p = Path(data_root)
    path_1 = data_root_p / "frame_0001.png"
    path_2 = data_root_p / "frame_0002.png"
    if not path_1.exists() or not path_2.exists():
        raise FileNotFoundError(f"Expected two frames at {path_1} and {path_2}")

    tfm = T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    x_t = tfm(Image.open(path_1).convert("RGB")).unsqueeze(0).to(device_t)
    x_tp1 = tfm(Image.open(path_2).convert("RGB")).unsqueeze(0).to(device_t)

    translator = UNetTranslator(image_size=image_size, base_channels=base_channels, bottleneck_channels=bottleneck_channels).to(device_t)
    opt = torch.optim.AdamW(translator.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    l1 = nn.L1Loss()

    translator.train()
    for epoch in range(1, epochs + 1):
        # Single pair training: predict residual then compose output
        r = translator(torch.cat([x_t, x_t], dim=1))
        y = torch.clamp(x_t + r, -1.0, 1.0)

        # Losses encouraging exact reconstruction
        loss_rec = l1(y, x_tp1)
        loss_edge = sobel_edge_loss(y, x_tp1)
        loss_ssim = ssim_loss_simple(y, x_tp1)
        loss = loss_rec + 0.1 * loss_edge + 0.5 * loss_ssim

        # Compute exact-match reward signal (non-differentiable; for logging/early stop)
        with torch.no_grad():
            y01 = _to01(y)
            t101 = _to01(x_tp1)
            y_u8 = (y01 * 255.0).round().to(torch.uint8)
            t1_u8 = (t101 * 255.0).round().to(torch.uint8)
            exact = bool(torch.equal(y_u8, t1_u8))
            pixel_acc = float((y_u8 == t1_u8).float().mean().item())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(translator.parameters(), 1.0)
        opt.step()

        tqdm.write(f"Epoch {epoch}/{epochs} | L1: {float(loss_rec):.4f} | Acc_exact: {pixel_acc:.4f} | Exact: {int(exact)}")

        # Save visualization each epoch
        translator.eval()
        with torch.no_grad():
            r_vis = translator(torch.cat([x_t, x_t], dim=1))
            y_vis = torch.clamp(x_t + r_vis, -1.0, 1.0)
        save_image_grid(x_t, y_vis, Path("samples/predictor_ae"), f"epoch_{epoch:03d}.png")
        translator.train()

        # Early stop when exact pixel match is achieved
        if exact:
            break

    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": translator.state_dict(), "config": {"image_size": image_size, "base_channels": base_channels, "bottleneck_channels": bottleneck_channels}}, out_path_p)
    return str(out_path_p)


@torch.no_grad()
def predict_next_ae(
    image_path: str,
    predictor_ckpt: str,
    out_path: str = "outputs_next/predictor_ae.png",
    device: str = "cuda",
) -> str:
    device_t = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    ckpt = torch.load(predictor_ckpt, map_location="cpu")
    cfg = ckpt["config"]  # type: ignore[index]
    translator = UNetTranslator(image_size=int(cfg["image_size"]), base_channels=int(cfg["base_channels"]), bottleneck_channels=int(cfg["bottleneck_channels"]))
    translator.load_state_dict(ckpt["state_dict"])  # type: ignore[index]
    translator = translator.to(device_t)
    translator.eval()

    img = Image.open(image_path).convert("RGB")
    tfm = T.Compose([
        T.Resize(int(cfg["image_size"]), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop(int(cfg["image_size"])),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    x = tfm(img).unsqueeze(0).to(device_t)
    r = translator(torch.cat([x, x], dim=1))
    y = torch.clamp(x + r, -1.0, 1.0)
    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)
    save_image_grid(x, y, out_path_p.parent, out_path_p.name)
    return str(out_path_p)


def eval_next_ae(
    data_root: str,
    predictor_ckpt: str,
    image_size: int = 128,
    batch_size: int = 4,
    num_batches: int = 10,
    device: str = "cuda",
) -> dict:
    device_t = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    ckpt = torch.load(predictor_ckpt, map_location="cpu")
    cfg = ckpt["config"]  # type: ignore[index]
    translator = UNetTranslator(image_size=int(cfg["image_size"]), base_channels=int(cfg["base_channels"]), bottleneck_channels=int(cfg["bottleneck_channels"]))
    translator.load_state_dict(ckpt["state_dict"])  # type: ignore[index]
    translator = translator.to(device_t)
    translator.eval()

    ds = FramePairDataset(root=data_root, image_size=image_size, is_train=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    psnr_vals: List[float] = []
    ssim_vals: List[float] = []
    processed = 0
    for x_t, x_tp1 in dl:
        x_t = x_t.to(device_t)
        x_tp1 = x_tp1.to(device_t)
        with torch.no_grad():
            r = translator(torch.cat([x_t, x_t], dim=1))
            y = torch.clamp(x_t + r, -1.0, 1.0)
        for i in range(x_t.shape[0]):
            psnr_vals.append(float(psnr_value(y[i:i+1], x_tp1[i:i+1]).item()))
            ssim_vals.append(float(ssim_value(y[i:i+1], x_tp1[i:i+1]).item()))
        processed += 1
        if processed >= num_batches:
            break
    return {"psnr": sum(psnr_vals)/max(1,len(psnr_vals)), "ssim": sum(ssim_vals)/max(1,len(ssim_vals))}


def train_editor(
    data_root: str,
    vqvae_ckpt: str,
    out_path: str,
    image_size: int = 128,
    batch_size: int = 8,
    epochs: int = 8,
    lr: float = 3e-4,
    embed_dim: int = 384,
    layers: int = 6,
    heads: int = 6,
    device: str = "cuda",
) -> Tuple[str, dict]:
    device_t = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    # Load VQ-VAE (frozen)
    ckpt = torch.load(vqvae_ckpt, map_location="cpu")
    vqcfg = VQVAEConfig(**ckpt["config"])  # type: ignore[arg-type]
    vqvae = VQVAE(vqcfg).to(device_t)
    vqvae.load_state_dict(ckpt["state_dict"])  # type: ignore[index]
    vqvae.eval()

    ds = SingleImageDataset(root=data_root, image_size=image_size, is_train=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    sample_x = next(iter(dl))
    with torch.no_grad():
        sample_indices, _ = vqvae.encode_to_indices(sample_x.to(device_t)[:1])
    h_code, w_code = sample_indices.shape[-2], sample_indices.shape[-1]
    tokens_per_frame = h_code * w_code

    num_embeddings = vqcfg.num_embeddings
    bos_id = num_embeddings
    sep_id = num_embeddings + 1  # start of generation
    mask0_id = num_embeddings + 2
    mask1_id = num_embeddings + 3
    instr_start_id = num_embeddings + 4
    num_instr = len(INSTRUCTION_NAMES)
    vocab_size = num_embeddings + 4 + num_instr

    # Sequence = [BOS] [INSTR] prev_tokens mask_tokens [SEP] target_tokens[:-1]
    cond_len = 1 + 1 + tokens_per_frame + tokens_per_frame + 1
    max_seq_len = cond_len + (tokens_per_frame - 1)

    gptcfg = GPTConfig(vocab_size=vocab_size, max_seq_len=max_seq_len, embed_dim=embed_dim, num_layers=layers, num_heads=heads, dropout=0.1)
    editor = GPTNextFrame(gptcfg).to(device_t)

    scaler = torch.amp.GradScaler(device_t.type)
    opt = torch.optim.AdamW(editor.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    last_batch_x = None
    for epoch in range(1, epochs + 1):
        editor.train()
        pbar = tqdm(dl, desc=f"Editor {epoch}/{epochs}")
        for x in pbar:
            x = x.to(device_t, non_blocking=True)
            last_batch_x = x
            b, c, h, w = x.shape

            # Make mask and instruction per-sample
            masks = []
            instr_ids = []
            for _ in range(b):
                masks.append(make_random_rect_mask(h, w))
                instr_ids.append(random.randint(0, num_instr - 1))
            mask01 = torch.stack(masks, dim=0).to(device_t)  # (B,1,H,W)
            instr_ids_t = torch.tensor(instr_ids, dtype=torch.long, device=device_t)

            # Build edited target
            x01 = torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)
            x_edit01_list = []
            for i in range(b):
                x_edit01_list.append(apply_instruction_masked(x01[i:i+1], mask01[i:i+1], int(instr_ids_t[i].item())))
            x_edit01 = torch.cat(x_edit01_list, dim=0)
            x_edit = torch.clamp(x_edit01 * 2.0 - 1.0, -1.0, 1.0)

            with torch.no_grad():
                ids_prev, _ = vqvae.encode_to_indices(x)
                ids_tgt, _ = vqvae.encode_to_indices(x_edit)

            bsz = ids_prev.shape[0]
            prev_tokens = ids_prev.view(bsz, -1)
            tgt_tokens = ids_tgt.view(bsz, -1)

            # Mask tokens in code grid resolution
            with torch.no_grad():
                mask_low = F.interpolate(mask01, size=(h_code, w_code), mode="nearest")
            mask_flat = mask_low.view(bsz, -1)
            mask_token_ids = torch.where(mask_flat > 0.5, torch.full_like(mask_flat, mask1_id), torch.full_like(mask_flat, mask0_id))

            bos = torch.full((bsz, 1), bos_id, dtype=torch.long, device=device_t)
            instr_tok = (instr_ids_t + instr_start_id).view(bsz, 1)
            sep = torch.full((bsz, 1), sep_id, dtype=torch.long, device=device_t)

            input_ids = torch.cat([bos, instr_tok, prev_tokens, mask_token_ids, sep, tgt_tokens[:, :-1]], dim=1)
            seg_cond = torch.zeros((bsz, cond_len), dtype=torch.long, device=device_t)
            seg_gen = torch.ones((bsz, tokens_per_frame - 1), dtype=torch.long, device=device_t)
            segment_ids = torch.cat([seg_cond, seg_gen], dim=1)

            with torch.amp.autocast(device_type=device_t.type, enabled=True):
                logits = editor(input_ids=input_ids, segment_ids=segment_ids)
                targets = torch.full_like(input_ids, fill_value=-100)
                targets[:, -tokens_per_frame:] = tgt_tokens
                loss = ce_loss(logits.view(-1, logits.size(-1)), targets.view(-1))

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(editor.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            pbar.set_postfix({"loss": float(loss)})

        # Qualitative sample
        if last_batch_x is not None:
            editor.eval()
            x_vis = last_batch_x[:1]
            b, _, h, w = x_vis.shape
            mask_vis = make_random_rect_mask(h, w).to(device_t).unsqueeze(0)
            instr_vis = random.randint(0, num_instr - 1)
            with torch.no_grad():
                ids_prev, _ = vqvae.encode_to_indices(x_vis)
                prev_tokens = ids_prev.view(1, -1)
                mask_low = F.interpolate(mask_vis, size=(h_code, w_code), mode="nearest")
                mask_flat = mask_low.view(1, -1)
                mask_token_ids = torch.where(mask_flat > 0.5, torch.full_like(mask_flat, mask1_id), torch.full_like(mask_flat, mask0_id))
                bos = torch.full((1, 1), bos_id, dtype=torch.long, device=device_t)
                instr_tok = torch.full((1, 1), instr_start_id + instr_vis, dtype=torch.long, device=device_t)
                sep = torch.full((1, 1), sep_id, dtype=torch.long, device=device_t)
                prefix_ids = torch.cat([bos, instr_tok, prev_tokens, mask_token_ids, sep], dim=1)
                seg_cond = torch.zeros((1, prefix_ids.shape[1]), dtype=torch.long, device=device_t)
                generated = editor.generate(prefix_ids, seg_cond, max_new_tokens=tokens_per_frame, temperature=0.9, top_k=50, forbidden_token_ids=(bos_id, sep_id))
                out_tokens = generated[:, -tokens_per_frame:]
                tokens_reshaped = out_tokens.view(1, h_code, w_code)
                x_rec = vqvae.decode_from_indices(tokens_reshaped)
            save_image_grid(x_vis, x_rec, Path("samples/editor"), f"epoch_{epoch:03d}.png")

    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "tokens_per_frame": tokens_per_frame,
        "h_code": h_code,
        "w_code": w_code,
        "bos_id": bos_id,
        "sep_id": sep_id,
        "mask0_id": mask0_id,
        "mask1_id": mask1_id,
        "instr_start_id": instr_start_id,
        "num_instr": num_instr,
        "num_embeddings": num_embeddings,
        "image_size": image_size,
    }
    torch.save({"state_dict": editor.state_dict(), "config": editor.cfg.__dict__, "meta": meta}, out_path_p)
    return str(out_path_p), meta


@torch.no_grad()
def edit_image_with_model(
    image_path: str,
    vqvae_ckpt: str,
    editor_ckpt: str,
    instruction: str,
    out_path: str = "outputs_edit/edited.png",
    device: str = "cuda",
) -> str:
    device_t = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    vq_ckpt = torch.load(vqvae_ckpt, map_location="cpu")
    vqcfg = VQVAEConfig(**vq_ckpt["config"])  # type: ignore[arg-type]
    vqvae = VQVAE(vqcfg).to(device_t)
    vqvae.load_state_dict(vq_ckpt["state_dict"])  # type: ignore[index]
    vqvae.eval()

    ed_ckpt = torch.load(editor_ckpt, map_location="cpu")
    edcfg = GPTConfig(**ed_ckpt["config"])  # type: ignore[arg-type]
    editor = GPTNextFrame(edcfg).to(device_t)
    editor.load_state_dict(ed_ckpt["state_dict"])  # type: ignore[index]
    editor.eval()

    meta = ed_ckpt["meta"]  # type: ignore[index]
    tokens_per_frame = int(meta["tokens_per_frame"])  # type: ignore[index]
    h_code = int(meta["h_code"])  # type: ignore[index]
    w_code = int(meta["w_code"])  # type: ignore[index]
    bos_id = int(meta["bos_id"])  # type: ignore[index]
    sep_id = int(meta["sep_id"])  # type: ignore[index]
    mask0_id = int(meta["mask0_id"])  # type: ignore[index]
    mask1_id = int(meta["mask1_id"])  # type: ignore[index]
    instr_start_id = int(meta["instr_start_id"])  # type: ignore[index]
    num_instr = int(meta["num_instr"])  # type: ignore[index]
    image_size = int(meta["image_size"])  # type: ignore[index]

    instr_id = instruction_name_to_id(instruction)

    img = Image.open(image_path).convert("RGB")
    tfm = T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    x = tfm(img).unsqueeze(0).to(device_t)
    ids_prev, _ = vqvae.encode_to_indices(x)
    prev_tokens = ids_prev.view(1, -1)

    # random rectangle mask at inference (can be later provided by user)
    _, _, h, w = x.shape
    mask = make_random_rect_mask(h, w).to(device_t).unsqueeze(0)
    mask_low = F.interpolate(mask, size=(h_code, w_code), mode="nearest")
    mask_flat = mask_low.view(1, -1)
    mask_token_ids = torch.where(mask_flat > 0.5, torch.full_like(mask_flat, mask1_id), torch.full_like(mask_flat, mask0_id))

    bos = torch.full((1, 1), bos_id, dtype=torch.long, device=device_t)
    instr_tok = torch.full((1, 1), instr_start_id + instr_id, dtype=torch.long, device=device_t)
    sep = torch.full((1, 1), sep_id, dtype=torch.long, device=device_t)
    prefix_ids = torch.cat([bos, instr_tok, prev_tokens, mask_token_ids, sep], dim=1)
    seg_cond = torch.zeros((1, prefix_ids.shape[1]), dtype=torch.long, device=device_t)
    generated = editor.generate(prefix_ids, seg_cond, max_new_tokens=tokens_per_frame, temperature=0.9, top_k=50, forbidden_token_ids=(bos_id, sep_id))
    out_tokens = generated[:, -tokens_per_frame:]
    tokens_reshaped = out_tokens.view(1, h_code, w_code)
    x_rec = vqvae.decode_from_indices(tokens_reshaped)

    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)
    save_image_grid(x, x_rec, out_path_p.parent, out_path_p.name)
    return str(out_path_p)


def train_editor_from_ae(
    data_root: str,
    ae_ckpt: str,
    out_path: str,
    image_size: int = 128,
    batch_size: int = 8,
    epochs: int = 8,
    lr: float = 1e-4,
    base_channels: int = 64,
    bottleneck_channels: int = 512,
    device: str = "cuda",
) -> str:
    device_t = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    # Load AE checkpoint (conv or unet)
    ckpt = torch.load(ae_ckpt, map_location="cpu")
    ae_cfg = AEConfig(**ckpt["config"])  # type: ignore[arg-type]
    arch = ckpt.get("arch", "conv")

    # Build editor
    ed_cfg = EditorUNetConfig(image_size=image_size, base_channels=base_channels, bottleneck_channels=bottleneck_channels, num_instructions=len(INSTRUCTION_NAMES))
    editor = UNetEditor(ed_cfg).to(device_t)

    # Initialize from AE weights if UNet-based
    if arch == "unet":
        ae_model = UNetAutoencoder(ae_cfg)
        ae_model.load_state_dict(ckpt["state_dict"])  # type: ignore[index]
        editor.init_from_unet_autoencoder(ae_model.state_dict())

    ds = SingleImageDataset(root=data_root, image_size=image_size, is_train=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    opt = torch.optim.Adam(editor.parameters(), lr=lr)
    l1 = nn.L1Loss()

    last_batch_x = None
    editor.train()
    for epoch in range(1, epochs + 1):
        pbar = tqdm(dl, desc=f"EditorUNet {epoch}/{epochs}")
        for x in pbar:
            x = x.to(device_t, non_blocking=True)
            last_batch_x = x
            b, c, h, w = x.shape
            masks = []
            instr_ids = []
            for _ in range(b):
                masks.append(make_random_edit_mask(h, w))
                instr_ids.append(random.randint(0, len(INSTRUCTION_NAMES) - 1))
            mask01 = torch.stack(masks, dim=0).to(device_t)
            instr_ids_t = torch.tensor(instr_ids, dtype=torch.long, device=device_t)

            x01 = torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)
            x_edit01_list = []
            for i in range(b):
                x_edit01_list.append(apply_instruction_masked(x01[i:i+1], mask01[i:i+1], int(instr_ids_t[i].item())))
            x_tgt = torch.clamp(torch.cat(x_edit01_list, dim=0) * 2.0 - 1.0, -1.0, 1.0)

            cond = build_editor_condition(mask01, instr_ids_t, h, w, len(INSTRUCTION_NAMES))
            inp = torch.cat([x, cond], dim=1)
            out = editor(inp)

            # Preserve outside-mask, edit inside-mask; supervise everywhere
            loss_rec = l1(out, x_tgt)
            loss_edge = sobel_edge_loss(out, x_tgt)
            loss = loss_rec + 0.1 * loss_edge
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(editor.parameters(), 1.0)
            opt.step()
            pbar.set_postfix({"l1": float(loss_rec)})

        # sample
        if last_batch_x is not None:
            editor.eval()
            x_vis = last_batch_x[:1]
            b, _, h, w = x_vis.shape
            mask_vis = make_random_edit_mask(h, w).to(device_t).unsqueeze(0)
            instr_vis = random.randint(0, len(INSTRUCTION_NAMES) - 1)
            cond_vis = build_editor_condition(mask_vis, torch.tensor([instr_vis], device=device_t), h, w, len(INSTRUCTION_NAMES))
            with torch.no_grad():
                out_vis = editor(torch.cat([x_vis, cond_vis], dim=1))
            save_image_grid(x_vis, out_vis, Path("samples/editor_unet"), f"epoch_{epoch:03d}.png")
            editor.train()

    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": editor.state_dict(), "config": ed_cfg.__dict__}, out_path_p)
    return str(out_path_p)


@torch.no_grad()
def edit_image_unet(
    image_path: str,
    editor_ckpt: str,
    instruction: str,
    out_path: str = "outputs_edit/edited_unet.png",
    device: str = "cuda",
) -> str:
    device_t = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    ckpt = torch.load(editor_ckpt, map_location="cpu")
    cfg = EditorUNetConfig(**ckpt["config"])  # type: ignore[arg-type]
    editor = UNetEditor(cfg).to(device_t)
    editor.load_state_dict(ckpt["state_dict"])  # type: ignore[index]
    editor.eval()

    img = Image.open(image_path).convert("RGB")
    tfm = T.Compose([
        T.Resize(cfg.image_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop(cfg.image_size),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    x = tfm(img).unsqueeze(0).to(device_t)
    _, _, h, w = x.shape
    mask = make_random_edit_mask(h, w).to(device_t).unsqueeze(0)
    instr_id = instruction_name_to_id(instruction)
    cond = build_editor_condition(mask, torch.tensor([instr_id], device=device_t), h, w, len(INSTRUCTION_NAMES))
    out = editor(torch.cat([x, cond], dim=1))

    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)
    save_image_grid(x, out, out_path_p.parent, out_path_p.name)
    return str(out_path_p)


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="All-in-one: train VQ-VAE, train GPT next-frame, and predict.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_ae = sub.add_parser("train_ae")
    p_ae.add_argument("--data-root", type=str, required=True)
    p_ae.add_argument("--out", type=str, default="checkpoints/ae.pt")
    p_ae.add_argument("--image-size", type=int, default=128)
    p_ae.add_argument("--batch-size", type=int, default=16)
    p_ae.add_argument("--epochs", type=int, default=20)
    p_ae.add_argument("--lr", type=float, default=1e-4)
    p_ae.add_argument("--ssim-weight", type=float, default=0.5)
    p_ae.add_argument("--base-channels", type=int, default=64)
    p_ae.add_argument("--bottleneck-channels", type=int, default=512)
    p_ae.add_argument("--use-unet", action="store_true")
    p_ae.add_argument("--edge-weight", type=float, default=0.0)
    p_ae.add_argument("--device", type=str, default="cuda")

    p_aep = sub.add_parser("reconstruct")
    p_aep.add_argument("--image", type=str, required=True)
    p_aep.add_argument("--ckpt", type=str, required=True)
    p_aep.add_argument("--out", type=str, default="outputs_ae/recon.png")
    p_aep.add_argument("--device", type=str, default="cuda")

    p_vq = sub.add_parser("train_vqvae")
    p_vq.add_argument("--data-root", type=str, required=True)
    p_vq.add_argument("--out", type=str, default="checkpoints/vqvae.pt")
    p_vq.add_argument("--image-size", type=int, default=128)
    p_vq.add_argument("--batch-size", type=int, default=8)
    p_vq.add_argument("--epochs", type=int, default=8)
    p_vq.add_argument("--lr", type=float, default=3e-4)
    p_vq.add_argument("--downsample-factor", type=int, default=16, choices=[4, 8, 16])
    p_vq.add_argument("--num-embeddings", type=int, default=512)
    p_vq.add_argument("--embedding-dim", type=int, default=128)
    p_vq.add_argument("--hidden-channels", type=int, default=96)
    p_vq.add_argument("--latent-channels", type=int, default=192)
    p_vq.add_argument("--commitment-cost", type=float, default=0.25)
    p_vq.add_argument("--ema-decay", type=float, default=0.99)
    p_vq.add_argument("--ema-warmup-steps", type=int, default=100)
    p_vq.add_argument("--max-codebook-norm", type=float, default=2.0)
    p_vq.add_argument("--dead-code-threshold", type=float, default=1.0)
    p_vq.add_argument("--dead-code-check-steps", type=int, default=100)
    p_vq.add_argument("--recon-weight", type=float, default=1.0)
    p_vq.add_argument("--device", type=str, default="cuda")

    p_gpt = sub.add_parser("train_gpt")
    p_gpt.add_argument("--data-root", type=str, required=True)
    p_gpt.add_argument("--vqvae", type=str, required=True)
    p_gpt.add_argument("--out", type=str, default="checkpoints/gpt_next.pt")
    p_gpt.add_argument("--image-size", type=int, default=128)
    p_gpt.add_argument("--batch-size", type=int, default=8)
    p_gpt.add_argument("--epochs", type=int, default=8)
    p_gpt.add_argument("--lr", type=float, default=3e-4)
    p_gpt.add_argument("--embed-dim", type=int, default=384)
    p_gpt.add_argument("--layers", type=int, default=6)
    p_gpt.add_argument("--heads", type=int, default=6)
    p_gpt.add_argument("--device", type=str, default="cuda")

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--image", type=str, required=True)
    p_pred.add_argument("--vqvae", type=str, required=True)
    p_pred.add_argument("--gpt", type=str, required=True)
    p_pred.add_argument("--steps", type=int, default=6)
    p_pred.add_argument("--temperature", type=float, default=0.9)
    p_pred.add_argument("--top-k", type=int, default=50)
    p_pred.add_argument("--out", type=str, default="outputs_next")
    p_pred.add_argument("--device", type=str, default="cuda")

    p_eval = sub.add_parser("eval_next")
    p_eval.add_argument("--data-root", type=str, required=True)
    p_eval.add_argument("--vqvae", type=str, required=True)
    p_eval.add_argument("--gpt", type=str, required=True)
    p_eval.add_argument("--num-batches", type=int, default=10)
    p_eval.add_argument("--image-size", type=int, default=128)
    p_eval.add_argument("--batch-size", type=int, default=4)
    p_eval.add_argument("--temperature", type=float, default=0.9)
    p_eval.add_argument("--top-k", type=int, default=50)
    p_eval.add_argument("--device", type=str, default="cuda")

    p_edit_train = sub.add_parser("train_editor")
    p_edit_train.add_argument("--data-root", type=str, required=True)
    p_edit_train.add_argument("--vqvae", type=str, required=True)
    p_edit_train.add_argument("--out", type=str, default="checkpoints/editor.pt")
    p_edit_train.add_argument("--image-size", type=int, default=128)
    p_edit_train.add_argument("--batch-size", type=int, default=8)
    p_edit_train.add_argument("--epochs", type=int, default=8)
    p_edit_train.add_argument("--lr", type=float, default=3e-4)
    p_edit_train.add_argument("--embed-dim", type=int, default=384)
    p_edit_train.add_argument("--layers", type=int, default=6)
    p_edit_train.add_argument("--heads", type=int, default=6)
    p_edit_train.add_argument("--device", type=str, default="cuda")

    p_edit = sub.add_parser("edit")
    p_edit.add_argument("--image", type=str, required=True)
    p_edit.add_argument("--vqvae", type=str, required=True)
    p_edit.add_argument("--editor", type=str, required=True)
    p_edit.add_argument("--instruction", type=str, required=True, choices=list(INSTRUCTION_NAMES))
    p_edit.add_argument("--out", type=str, default="outputs_edit/edited.png")
    p_edit.add_argument("--device", type=str, default="cuda")

    p_edit_unet_train = sub.add_parser("train_editor_from_ae")
    p_edit_unet_train.add_argument("--data-root", type=str, required=True)
    p_edit_unet_train.add_argument("--ae", type=str, required=True)
    p_edit_unet_train.add_argument("--out", type=str, default="checkpoints/editor_unet.pt")
    p_edit_unet_train.add_argument("--image-size", type=int, default=128)
    p_edit_unet_train.add_argument("--batch-size", type=int, default=8)
    p_edit_unet_train.add_argument("--epochs", type=int, default=8)
    p_edit_unet_train.add_argument("--lr", type=float, default=1e-4)
    p_edit_unet_train.add_argument("--base-channels", type=int, default=64)
    p_edit_unet_train.add_argument("--bottleneck-channels", type=int, default=512)
    p_edit_unet_train.add_argument("--device", type=str, default="cuda")

    p_edit_unet = sub.add_parser("edit_unet")
    p_edit_unet.add_argument("--image", type=str, required=True)
    p_edit_unet.add_argument("--editor", type=str, required=True)
    p_edit_unet.add_argument("--instruction", type=str, required=True, choices=list(INSTRUCTION_NAMES))
    p_edit_unet.add_argument("--out", type=str, default="outputs_edit/edited_unet.png")
    p_edit_unet.add_argument("--device", type=str, default="cuda")

    p_pred_ae_train = sub.add_parser("train_predictor_ae")
    p_pred_ae_train.add_argument("--data-root", type=str, required=True)
    p_pred_ae_train.add_argument("--ae", type=str, required=True)
    p_pred_ae_train.add_argument("--out", type=str, default="checkpoints/predictor_ae.pt")
    p_pred_ae_train.add_argument("--image-size", type=int, default=128)
    p_pred_ae_train.add_argument("--batch-size", type=int, default=8)
    p_pred_ae_train.add_argument("--epochs", type=int, default=8)
    p_pred_ae_train.add_argument("--lr", type=float, default=1e-4)
    p_pred_ae_train.add_argument("--base-channels", type=int, default=64)
    p_pred_ae_train.add_argument("--bottleneck-channels", type=int, default=512)
    p_pred_ae_train.add_argument("--device", type=str, default="cuda")

    p_pred_ae = sub.add_parser("predict_next_ae")
    p_pred_ae.add_argument("--image", type=str, required=True)
    p_pred_ae.add_argument("--predictor", type=str, required=True)
    p_pred_ae.add_argument("--out", type=str, default="outputs_next/predictor_ae.png")
    p_pred_ae.add_argument("--device", type=str, default="cuda")

    p_eval_ae = sub.add_parser("eval_next_ae")
    p_eval_ae.add_argument("--data-root", type=str, required=True)
    p_eval_ae.add_argument("--predictor", type=str, required=True)
    p_eval_ae.add_argument("--image-size", type=int, default=128)
    p_eval_ae.add_argument("--batch-size", type=int, default=4)
    p_eval_ae.add_argument("--num-batches", type=int, default=10)
    p_eval_ae.add_argument("--device", type=str, default="cuda")

    return p.parse_args()


def main() -> None:
    args = parse_cli()
    if args.cmd == "train_ae":
        ckpt = train_autoencoder_phase1(
            data_root=args.data_root,
            out_path=args.out,
            image_size=args.image_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            ssim_weight=args.ssim_weight,
            base_channels=args.base_channels,
            bottleneck_channels=args.bottleneck_channels,
            use_unet=args.use_unet,
            edge_weight=args.edge_weight,
            device=args.device,
        )
        print(f"Saved AE: {ckpt}")
    elif args.cmd == "reconstruct":
        out = reconstruct_image_phase1(image_path=args.image, ckpt_path=args.ckpt, out_path=args.out, device=args.device)
        print(f"Saved reconstruction to {out}")
    elif args.cmd == "train_vqvae":
        ckpt = train_vqvae(
            data_root=args.data_root,
            out_path=args.out,
            image_size=args.image_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            downsample_factor=args.downsample_factor,
            num_embeddings=args.num_embeddings,
            embedding_dim=args.embedding_dim,
            hidden_channels=args.hidden_channels,
            latent_channels=args.latent_channels,
            commitment_cost=args.commitment_cost,
            ema_decay=args.ema_decay,
            ema_warmup_steps=args.ema_warmup_steps,
            max_codebook_norm=args.max_codebook_norm,
            dead_code_threshold=args.dead_code_threshold,
            dead_code_check_steps=args.dead_code_check_steps,
            recon_weight=args.recon_weight,
            device=args.device,
        )
        print(f"Saved VQ-VAE: {ckpt}")
    elif args.cmd == "train_gpt":
        ckpt, meta = train_gpt_next(data_root=args.data_root, vqvae_ckpt=args.vqvae, out_path=args.out, image_size=args.image_size, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, embed_dim=args.embed_dim, layers=args.layers, heads=args.heads, device=args.device)
        print(f"Saved GPT: {ckpt}")
    elif args.cmd == "predict":
        predict_autoregressive(image_path=args.image, vqvae_ckpt=args.vqvae, gpt_ckpt=args.gpt, steps=args.steps, temperature=args.temperature, top_k=args.top_k, out_dir=args.out, device=args.device)
        print(f"Saved predicted frames to {args.out}")
    elif args.cmd == "eval_next":
        metrics = evaluate_next_frame(
            data_root=args.data_root,
            vqvae_ckpt=args.vqvae,
            gpt_ckpt=args.gpt,
            num_batches=args.num_batches,
            image_size=args.image_size,
            batch_size=args.batch_size,
            temperature=args.temperature,
            top_k=args.top_k,
            device=args.device,
        )
        print({"psnr": round(metrics["psnr"], 3), "ssim": round(metrics["ssim"], 4)})
    elif args.cmd == "train_editor":
        ckpt, meta = train_editor(
            data_root=args.data_root,
            vqvae_ckpt=args.vqvae,
            out_path=args.out,
            image_size=args.image_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            embed_dim=args.embed_dim,
            layers=args.layers,
            heads=args.heads,
            device=args.device,
        )
        print(f"Saved Editor: {ckpt}")
    elif args.cmd == "edit":
        out = edit_image_with_model(
            image_path=args.image,
            vqvae_ckpt=args.vqvae,
            editor_ckpt=args.editor,
            instruction=args.instruction,
            out_path=args.out,
            device=args.device,
        )
        print(f"Saved edited image to {out}")
    elif args.cmd == "train_editor_from_ae":
        ckpt = train_editor_from_ae(
            data_root=args.data_root,
            ae_ckpt=args.ae,
            out_path=args.out,
            image_size=args.image_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            base_channels=args.base_channels,
            bottleneck_channels=args.bottleneck_channels,
            device=args.device,
        )
        print(f"Saved UNet Editor: {ckpt}")
    elif args.cmd == "edit_unet":
        out = edit_image_unet(
            image_path=args.image,
            editor_ckpt=args.editor,
            instruction=args.instruction,
            out_path=args.out,
            device=args.device,
        )
        print(f"Saved edited image to {out}")
    elif args.cmd == "train_predictor_ae":
        ckpt = train_predictor_ae(
            data_root=args.data_root,
            ae_ckpt=args.ae,
            out_path=args.out,
            image_size=args.image_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            base_channels=args.base_channels,
            bottleneck_channels=args.bottleneck_channels,
            device=args.device,
        )
        print(f"Saved PredictorAE: {ckpt}")
    elif args.cmd == "predict_next_ae":
        out = predict_next_ae(
            image_path=args.image,
            predictor_ckpt=args.predictor,
            out_path=args.out,
            device=args.device,
        )
        print(f"Saved predicted frame to {out}")
    elif args.cmd == "eval_next_ae":
        metrics = eval_next_ae(
            data_root=args.data_root,
            predictor_ckpt=args.predictor,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_batches=args.num_batches,
            device=args.device,
        )
        print({"psnr": round(metrics["psnr"], 3), "ssim": round(metrics["ssim"], 4)})
    elif args.cmd == "train_editor":
        ckpt, meta = train_editor(
            data_root=args.data_root,
            vqvae_ckpt=args.vqvae,
            out_path=args.out,
            image_size=args.image_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            embed_dim=args.embed_dim,
            layers=args.layers,
            heads=args.heads,
            device=args.device,
        )
        print(f"Saved Editor: {ckpt}")
    elif args.cmd == "edit":
        out = edit_image_with_model(
            image_path=args.image,
            vqvae_ckpt=args.vqvae,
            editor_ckpt=args.editor,
            instruction=args.instruction,
            out_path=args.out,
            device=args.device,
        )
        print(f"Saved edited image to {out}")


if __name__ == "__main__":
    main()
# Legacy demo code removed to avoid ModuleNotFoundError and duplicate entry points.