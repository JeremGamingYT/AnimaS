import argparse
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
import torchvision.transforms as T

from models.vqvae import VQVAE, VQVAEConfig
from models.gpt_next_frame import GPTNextFrame, GPTConfig


def load_image(path: Path, image_size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    tfm = T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return tfm(img).unsqueeze(0)


@torch.no_grad()
def decode_tokens_to_image(vqvae: VQVAE, token_ids: torch.Tensor, h_code: int, w_code: int) -> Image.Image:
    b = token_ids.shape[0]
    tokens = token_ids.view(b, h_code, w_code)
    x_rec = vqvae.decode_from_indices(tokens)
    x_rec = torch.clamp((x_rec + 1.0) * 0.5, 0.0, 1.0)
    img = T.ToPILImage()(x_rec[0].cpu())
    return img


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict next frames autoregressively from a single image")
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--vqvae", type=str, required=True)
    p.add_argument("--gpt", type=str, required=True)
    p.add_argument("--steps", type=int, default=1, help="Number of next frames to generate")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--out", type=str, default="outputs_next")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # Load VQ-VAE
    vq_ckpt = torch.load(args.vqvae, map_location="cpu")
    vqcfg = VQVAEConfig(**vq_ckpt["config"])  # type: ignore[arg-type]
    vqvae = VQVAE(vqcfg).to(device)
    vqvae.load_state_dict(vq_ckpt["state_dict"])  # type: ignore[index]
    vqvae.eval()

    # Load GPT
    gpt_ckpt = torch.load(args.gpt, map_location="cpu")
    gptcfg = GPTConfig(**gpt_ckpt["config"])  # type: ignore[arg-type]
    gpt = GPTNextFrame(gptcfg).to(device)
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

    # Prepare input image
    x0 = load_image(Path(args.image), image_size).to(device)
    with torch.no_grad():
        prev_ids, _ = vqvae.encode_to_indices(x0)
    b = prev_ids.shape[0]
    prev_tokens = prev_ids.view(b, -1)

    vocab_size = num_embeddings + 2

    outputs = []
    current_prev = prev_tokens
    for step in range(args.steps):
        bos = torch.full((b, 1), bos_id, dtype=torch.long, device=device)
        sep = torch.full((b, 1), sep_id, dtype=torch.long, device=device)
        prefix_ids = torch.cat([bos, current_prev, sep], dim=1)
        # Segment ids must match training: BOS+prev=0, SEP=1
        seg_prev = torch.zeros((b, 1 + current_prev.shape[1]), dtype=torch.long, device=device)
        seg_sep = torch.ones((b, 1), dtype=torch.long, device=device)
        prefix_segs = torch.cat([seg_prev, seg_sep], dim=1)
        generated = gpt.generate(
            prefix_ids=prefix_ids,
            prefix_segs=prefix_segs,
            max_new_tokens=tokens_per_frame,
            temperature=args.temperature,
            top_k=args.top_k,
            forbidden_token_ids=(bos_id, sep_id),
        )
        next_tokens = generated[:, -tokens_per_frame:]
        img = decode_tokens_to_image(vqvae, next_tokens, h_code, w_code)
        outputs.append(img)
        # Autoregressive: feed next as prev
        current_prev = next_tokens

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(outputs):
        img.save(out_dir / f"next_{i:03d}.png")
    print(f"Saved {len(outputs)} frames to {out_dir}")


if __name__ == "__main__":
    main()


