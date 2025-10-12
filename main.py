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
import os
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm


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
        x_t = self.transform(img_t)
        x_tp1 = self.transform(img_tp1)
        return x_t, x_tp1


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
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25, use_ema: bool = True, ema_decay: float = 0.99, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.commitment_cost = float(commitment_cost)
        self.use_ema = bool(use_ema)
        self.ema_decay = float(ema_decay)
        self.eps = float(eps)
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / self.num_embeddings, 1.0 / self.num_embeddings)
        if self.use_ema:
            self.register_buffer("ema_cluster_size", torch.zeros(self.num_embeddings))
            self.register_buffer("ema_w", self.embedding.weight.data.clone())

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Compute quantization in float32 to avoid NaN/Inf under AMP/FP16
        orig_dtype = z_e.dtype
        b, c, h, w = z_e.shape
        z_e_32 = z_e.float()
        z_e_flat = z_e_32.permute(0, 2, 3, 1).contiguous().view(-1, c)
        emb_w = self.embedding.weight.float()

        distances = (
            z_e_flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * z_e_flat @ emb_w.t()
            + emb_w.pow(2).sum(dim=1)
        )
        encoding_indices = torch.argmin(distances, dim=1)
        z_q_32 = F.embedding(encoding_indices, emb_w).view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

        commitment = self.commitment_cost * F.mse_loss(z_e_32.detach(), z_q_32)
        if self.use_ema and self.training:
            encodings_one_hot = F.one_hot(encoding_indices, self.num_embeddings).to(z_e_flat.dtype)
            cluster_size = encodings_one_hot.sum(dim=0)
            self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size, alpha=1 - self.ema_decay)
            n = self.ema_cluster_size.sum()
            cluster_size = ((self.ema_cluster_size + self.eps) / (n + self.num_embeddings * self.eps)) * n
            embed_sum = encodings_one_hot.t() @ z_e_flat
            self.ema_w.mul_(self.ema_decay).add_(embed_sum, alpha=1 - self.ema_decay)
            self.embedding.weight.data.copy_(self.ema_w / cluster_size.unsqueeze(1))
            codebook = torch.tensor(0.0, device=z_e.device, dtype=z_e_32.dtype)
        else:
            codebook = F.mse_loss(z_e_32, z_q_32.detach())
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


class GPTNextFrame(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.embed_dim)
        self.seg_embed = nn.Embedding(2, cfg.embed_dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([GPTBlock(cfg.embed_dim, cfg.num_heads, dropout=cfg.dropout) for _ in range(cfg.num_layers)])
        self.ln_f = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
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
    def generate(self, prefix_ids: torch.Tensor, prefix_segs: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = 50, forbidden_token_ids: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
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


def train_vqvae(data_root: str, out_path: str, image_size: int = 128, batch_size: int = 8, epochs: int = 8, lr: float = 3e-4, downsample_factor: int = 16, num_embeddings: int = 512, embedding_dim: int = 128, hidden_channels: int = 96, latent_channels: int = 192, device: str = "cuda") -> str:
    device_t = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    ds = FramePairDataset(root=data_root, image_size=image_size, is_train=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    cfg = VQVAEConfig(image_size=image_size, in_channels=3, hidden_channels=hidden_channels, latent_channels=latent_channels, num_embeddings=num_embeddings, embedding_dim=embedding_dim, downsample_factor=downsample_factor, commitment_cost=0.25, use_ema=True, ema_decay=0.99)
    model = VQVAE(cfg).to(device_t)
    scaler = torch.amp.GradScaler(device_t.type)
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
                loss = recon_loss + vq_loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            pbar.set_postfix({"recon": float(recon_loss), "vq": float(vq_loss)})

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
    gptcfg = GPTConfig(vocab_size=vocab_size, max_seq_len=max_seq_len, embed_dim=embed_dim, num_layers=layers, num_heads=heads, dropout=0.1)
    gpt = GPTNextFrame(gptcfg).to(device_t)

    scaler = torch.amp.GradScaler(device_t.type)
    opt = torch.optim.AdamW(gpt.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
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

            with torch.amp.autocast(device_type=device_t.type, enabled=True):
                logits = gpt(input_ids=input_ids, segment_ids=segment_ids)
                targets = torch.full_like(input_ids, fill_value=-100)
                targets[:, -tokens_per_frame:] = next_tokens
                loss = ce_loss(logits.view(-1, logits.size(-1)), targets.view(-1))

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
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
                generated = gpt.generate(prefix_ids, prefix_segs, max_new_tokens=tokens_per_frame, temperature=0.9, top_k=50, forbidden_token_ids=(bos_id, sep_id))
                next_tokens = generated[:, -tokens_per_frame:]
                tokens_reshaped = next_tokens.view(1, h_code, w_code)
                x_rec = vqvae.decode_from_indices(tokens_reshaped)
            save_image_grid(x_vis_t, x_rec, Path("samples/next_frame"), f"epoch_{epoch:03d}.png")

    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)
    meta = {"tokens_per_frame": tokens_per_frame, "h_code": h_code, "w_code": w_code, "bos_id": bos_id, "sep_id": sep_id, "num_embeddings": num_embeddings, "image_size": image_size}
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


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="All-in-one: train VQ-VAE, train GPT next-frame, and predict.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_vq = sub.add_parser("train_vqvae")
    p_vq.add_argument("--data-root", type=str, required=True)
    p_vq.add_argument("--out", type=str, default="checkpoints/vqvae.pt")
    p_vq.add_argument("--image-size", type=int, default=128)
    p_vq.add_argument("--batch-size", type=int, default=8)
    p_vq.add_argument("--epochs", type=int, default=8)
    p_vq.add_argument("--lr", type=float, default=3e-4)
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

    return p.parse_args()


def main() -> None:
    args = parse_cli()
    if args.cmd == "train_vqvae":
        ckpt = train_vqvae(data_root=args.data_root, out_path=args.out, image_size=args.image_size, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, device=args.device)
        print(f"Saved VQ-VAE: {ckpt}")
    elif args.cmd == "train_gpt":
        ckpt, meta = train_gpt_next(data_root=args.data_root, vqvae_ckpt=args.vqvae, out_path=args.out, image_size=args.image_size, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, embed_dim=args.embed_dim, layers=args.layers, heads=args.heads, device=args.device)
        print(f"Saved GPT: {ckpt}")
    elif args.cmd == "predict":
        predict_autoregressive(image_path=args.image, vqvae_ckpt=args.vqvae, gpt_ckpt=args.gpt, steps=args.steps, temperature=args.temperature, top_k=args.top_k, out_dir=args.out, device=args.device)
        print(f"Saved predicted frames to {args.out}")


if __name__ == "__main__":
    main()
# Legacy demo code removed to avoid ModuleNotFoundError and duplicate entry points.