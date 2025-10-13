import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T

from data.next_frame_dataset import FramePairDataset, default_transforms
from models.vqvae import VQVAE, VQVAEConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train VQ-VAE tokenizer on anime frames")
    p.add_argument("--data-root", type=str, required=True, help="Path to frames root (folder or subfolders)")
    p.add_argument("--sequence-by-subfolder", action="store_true", help="Treat each subfolder as a sequence")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num-embeddings", type=int, default=512)
    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--hidden-channels", type=int, default=96)
    p.add_argument("--latent-channels", type=int, default=192)
    p.add_argument("--downsample-factor", type=int, default=16, choices=[4, 8, 16])
    p.add_argument("--commitment-cost", type=float, default=0.25)
    p.add_argument("--out", type=str, default="checkpoints/vqvae.pt")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    train_ds = FramePairDataset(
        root=args.data_root,
        image_size=args.image_size,
        is_train=True,
        sequence_by_subfolder=args.sequence_by_subfolder,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    cfg = VQVAEConfig(
        image_size=args.image_size,
        in_channels=3,
        hidden_channels=args.hidden_channels,
        latent_channels=args.latent_channels,
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
        downsample_factor=args.downsample_factor,
        commitment_cost=args.commitment_cost,
    )
    model = VQVAE(cfg).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.device.startswith("cuda") and torch.cuda.is_available()))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    recon_crit = nn.L1Loss()

    model.train()
    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        running_recon = 0.0
        running_vq = 0.0
        for batch in pbar:
            x_t, x_tp1 = batch
            x = x_t.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
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

            running_recon += float(recon_loss.item())
            running_vq += float(vq_loss.item())
            pbar.set_postfix({"recon": running_recon / (pbar.n + 1), "vq": running_vq / (pbar.n + 1)})

        # Save sample reconstruction at epoch end
        model.eval()
        with torch.no_grad():
            x_vis = x[:4].to(device)
            out_vis = model(x_vis)
            rec = torch.clamp((out_vis["recon"] + 1.0) * 0.5, 0.0, 1.0)
        grid = T.ToPILImage()(torch.cat([x_vis.cpu(), rec.cpu()], dim=3)[0])
        samples_dir = Path("samples/vqvae")
        samples_dir.mkdir(parents=True, exist_ok=True)
        grid.save(samples_dir / f"epoch_{epoch:03d}.png")
        model.train()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": model.config.__dict__,
    }, out_path)
    print(f"Saved VQ-VAE to {out_path}")


if __name__ == "__main__":
    main()


