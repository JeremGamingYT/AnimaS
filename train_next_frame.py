import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T

from data.next_frame_dataset import FramePairDataset
from models.vqvae import VQVAE, VQVAEConfig
from models.gpt_next_frame import GPTNextFrame, GPTConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GPT-like next-frame predictor on VQ-VAE tokens")
    p.add_argument("--data-root", type=str, required=True, help="Path to frames root (folder or subfolders)")
    p.add_argument("--sequence-by-subfolder", action="store_true", help="Treat each subfolder as a sequence")
    p.add_argument("--image-size", type=int, default=256, help="Must match VQ-VAE training size")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--vqvae", type=str, required=True, help="Path to trained VQ-VAE checkpoint")
    p.add_argument("--embed-dim", type=int, default=384)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--out", type=str, default="checkpoints/gpt_next_frame.pt")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # Load VQ-VAE
    ckpt = torch.load(args.vqvae, map_location="cpu")
    vqcfg = VQVAEConfig(**ckpt["config"])  # type: ignore[arg-type]
    vqvae = VQVAE(vqcfg).to(device)
    vqvae.load_state_dict(ckpt["state_dict"])  # type: ignore[index]
    vqvae.eval()

    # Data
    train_ds = FramePairDataset(
        root=args.data_root,
        image_size=args.image_size,
        is_train=True,
        sequence_by_subfolder=args.sequence_by_subfolder,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    # Compute token grid size using one sample
    sample_x, _ = next(iter(train_loader))
    with torch.no_grad():
        sample_indices, _ = vqvae.encode_to_indices(sample_x.to(device)[:1])
    h_code, w_code = sample_indices.shape[-2], sample_indices.shape[-1]
    tokens_per_frame = h_code * w_code

    num_embeddings = vqcfg.num_embeddings
    bos_id = num_embeddings  # special BOS
    sep_id = num_embeddings + 1  # special SEP
    vocab_size = num_embeddings + 2

    max_seq_len = 1 + tokens_per_frame + 1 + tokens_per_frame
    gptcfg = GPTConfig(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        embed_dim=args.embed_dim,
        num_layers=args.layers,
        num_heads=args.heads,
        dropout=args.dropout,
    )
    gpt = GPTNextFrame(gptcfg).to(device)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.device.startswith("cuda") and torch.cuda.is_available()))
    opt = torch.optim.AdamW(gpt.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(1, args.epochs + 1):
        gpt.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        running = 0.0
        for x_t, x_tp1 in pbar:
            x_t = x_t.to(device, non_blocking=True)
            x_tp1 = x_tp1.to(device, non_blocking=True)

            with torch.no_grad():
                ids_t, _ = vqvae.encode_to_indices(x_t)
                ids_tp1, _ = vqvae.encode_to_indices(x_tp1)
            b = ids_t.shape[0]
            prev_tokens = ids_t.view(b, -1)
            next_tokens = ids_tp1.view(b, -1)

            # Build sequence: [BOS] prev [SEP] next[:-1]
            bos = torch.full((b, 1), bos_id, dtype=torch.long, device=device)
            sep = torch.full((b, 1), sep_id, dtype=torch.long, device=device)
            input_ids = torch.cat([bos, prev_tokens, sep, next_tokens[:, :-1]], dim=1)
            # Segment ids: prev tokens=0 (including BOS and SEP?), choose BOS=0, prev=0, SEP=1, next=1
            seg_prev = torch.zeros((b, 1 + tokens_per_frame), dtype=torch.long, device=device)
            seg_next = torch.ones((b, 1 + tokens_per_frame - 1), dtype=torch.long, device=device)
            segment_ids = torch.cat([seg_prev, seg_next], dim=1)

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = gpt(input_ids=input_ids, segment_ids=segment_ids)
                # Targets: ignore all positions except last tokens_per_frame (for next tokens)
                targets = torch.full_like(input_ids, fill_value=-100)
                targets[:, -tokens_per_frame:] = next_tokens
                loss = ce_loss(logits.view(-1, logits.size(-1)), targets.view(-1))

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            running += float(loss.item())
            pbar.set_postfix({"loss": running / (pbar.n + 1)})

        # Save qualitative next-frame samples at epoch end
        gpt.eval()
        with torch.no_grad():
            x_vis_t = x_t[:1].to(device)
            ids_t, _ = vqvae.encode_to_indices(x_vis_t)
            prev_tokens = ids_t.view(1, -1)
            bos = torch.full((1, 1), vqcfg.num_embeddings, dtype=torch.long, device=device)
            sep = torch.full((1, 1), vqcfg.num_embeddings + 1, dtype=torch.long, device=device)
            prefix_ids = torch.cat([bos, prev_tokens, sep], dim=1)
            seg_prev = torch.zeros((1, 1 + prev_tokens.shape[1]), dtype=torch.long, device=device)
            seg_sep = torch.ones((1, 1), dtype=torch.long, device=device)
            prefix_segs = torch.cat([seg_prev, seg_sep], dim=1)
            generated = gpt.generate(prefix_ids, prefix_segs, max_new_tokens=tokens_per_frame, temperature=0.9, top_k=50, forbidden_token_ids=(vqcfg.num_embeddings, vqcfg.num_embeddings + 1))
            next_tokens = generated[:, -tokens_per_frame:]
            tokens_reshaped = next_tokens.view(1, h_code, w_code)
            x_rec = vqvae.decode_from_indices(tokens_reshaped)
            x_rec = torch.clamp((x_rec + 1.0) * 0.5, 0.0, 1.0)
        side_by_side = torch.cat([x_vis_t.cpu(), x_rec.cpu()], dim=3)[0]
        samples_dir = Path("samples/next_frame")
        samples_dir.mkdir(parents=True, exist_ok=True)
        T.ToPILImage()(side_by_side).save(samples_dir / f"epoch_{epoch:03d}.png")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": gpt.state_dict(),
        "config": gpt.cfg.__dict__,
        "meta": {
            "tokens_per_frame": tokens_per_frame,
            "h_code": h_code,
            "w_code": w_code,
            "bos_id": bos_id,
            "sep_id": sep_id,
            "num_embeddings": num_embeddings,
            "image_size": args.image_size,
        },
    }, out_path)
    print(f"Saved GPT next-frame model to {out_path}")


if __name__ == "__main__":
    main()