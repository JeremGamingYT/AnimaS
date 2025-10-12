from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, in_channels: int = 3, hidden_channels: int = 128, latent_channels: int = 256, downsample_factor: int = 16) -> None:
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
    def __init__(self, out_channels: int = 3, hidden_channels: int = 128, latent_channels: int = 256, downsample_factor: int = 16) -> None:
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
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25, use_ema: bool = False, ema_decay: float = 0.99, eps: float = 1e-5) -> None:
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
        b, c, h, w = z_e.shape
        z_e_flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, c)
        distances = (
            z_e_flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * z_e_flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1)
        )
        encoding_indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(encoding_indices).view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

        # Losses and EMA updates
        commitment = self.commitment_cost * F.mse_loss(z_e.detach(), z_q)
        if self.use_ema and self.training:
            encodings_one_hot = F.one_hot(encoding_indices, self.num_embeddings).type_as(z_e_flat)
            cluster_size = encodings_one_hot.sum(dim=0)
            self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size, alpha=1 - self.ema_decay)
            n = self.ema_cluster_size.sum()
            cluster_size = ((self.ema_cluster_size + self.eps) / (n + self.num_embeddings * self.eps)) * n
            embed_sum = encodings_one_hot.t() @ z_e_flat
            self.ema_w.mul_(self.ema_decay).add_(embed_sum, alpha=1 - self.ema_decay)
            self.embedding.weight.data.copy_(self.ema_w / cluster_size.unsqueeze(1))
            codebook = torch.tensor(0.0, device=z_e.device, dtype=z_e.dtype)
        else:
            codebook = F.mse_loss(z_e, z_q.detach())
        vq_loss = commitment + codebook

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        # Perplexity
        encodings_one_hot = F.one_hot(encoding_indices, self.num_embeddings).float()
        avg_probs = encodings_one_hot.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        indices = encoding_indices.view(b, h, w)
        return z_q_st, vq_loss, perplexity, indices

    def codes_to_embeddings(self, indices: torch.Tensor) -> torch.Tensor:
        b, h, w = indices.shape
        z_q = self.embedding(indices.view(b, -1)).view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return z_q


@dataclass
class VQVAEConfig:
    image_size: int = 256
    in_channels: int = 3
    hidden_channels: int = 128
    latent_channels: int = 256
    num_embeddings: int = 1024
    embedding_dim: int = 256
    downsample_factor: int = 16
    commitment_cost: float = 0.25
    use_ema: bool = False
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

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_e = self.encoder(x)
        z_e = self.proj_to_code(z_e)
        z_q, vq_loss, perplexity, indices = self.quantizer(z_e)
        z = self.proj_from_code(z_q)
        x_rec = self.decoder(z)
        return {
            "recon": x_rec,
            "vq_loss": vq_loss,
            "perplexity": perplexity,
            "indices": indices,
        }

    def code_grid_size(self, image_size: int) -> int:
        return image_size // self.config.downsample_factor


