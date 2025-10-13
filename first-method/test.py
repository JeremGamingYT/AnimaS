import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import torch
except ImportError as e:
    print("[ERROR] Missing dependency: torch. Install with: pip install torch --index-url https://download.pytorch.org/whl/cu121 (for CUDA) or pip install torch", file=sys.stderr)
    raise

try:
    from diffusers import StableVideoDiffusionPipeline
except ImportError as e:
    print("[ERROR] Missing dependency: diffusers. Install with: pip install diffusers transformers accelerate safetensors", file=sys.stderr)
    raise

try:
    from PIL import Image
except ImportError as e:
    print("[ERROR] Missing dependency: Pillow. Install with: pip install pillow", file=sys.stderr)
    raise

try:
    import imageio
except ImportError:
    imageio = None

try:
    import numpy as np
except ImportError as e:
    print("[ERROR] Missing dependency: numpy. Install with: pip install numpy", file=sys.stderr)
    raise


def select_device_and_dtype(preferred_device: Optional[str], prefer_half_precision: bool) -> Tuple[str, torch.dtype, bool, Optional[str]]:
    """Select device and dtype based on availability and user preference.

    Returns (device_str, dtype, use_half_precision, hf_variant).
    """
    if preferred_device is None or preferred_device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    else:
        device_str = preferred_device

    if device_str == "cuda" and torch.cuda.is_available() and prefer_half_precision:
        return device_str, torch.float16, True, "fp16"

    # MPS float16 support can be finicky; default to float32 unless explicitly requested
    if device_str == "mps" and prefer_half_precision:
        return device_str, torch.float16, True, None

    return device_str, torch.float32, False, None


def load_pipeline(
    model_id: str,
    device_str: str,
    dtype: torch.dtype,
    hf_variant: Optional[str],
    enable_attention_slicing: bool,
    enable_vae_slicing: bool,
    enable_cpu_offload: bool,
    debug: bool,
) -> StableVideoDiffusionPipeline:
    """Load the Stable Video Diffusion image-to-video pipeline configured for the given device/dtype."""
    load_kwargs = {"torch_dtype": dtype}
    if hf_variant is not None:
        load_kwargs["variant"] = hf_variant

    if debug:
        print(f"[INFO] Loading pipeline: {model_id} (dtype={dtype}, variant={hf_variant})")

    pipe = StableVideoDiffusionPipeline.from_pretrained(model_id, **load_kwargs)

    if enable_attention_slicing:
        try:
            pipe.enable_attention_slicing()
            if debug:
                print("[INFO] Attention slicing enabled")
        except Exception as e:
            if debug:
                print(f"[WARN] Attention slicing not supported: {e}")

    if enable_vae_slicing:
        try:
            pipe.enable_vae_slicing()
            if debug:
                print("[INFO] VAE slicing enabled")
        except Exception as e:
            if debug:
                print(f"[WARN] VAE slicing not supported: {e}")

    # Move to device or enable offload
    if device_str == "cuda" or device_str == "mps":
        pipe = pipe.to(device_str)
        if debug:
            print(f"[INFO] Pipeline moved to device: {device_str}")
        if enable_cpu_offload:
            try:
                pipe.enable_model_cpu_offload()
                if debug:
                    print("[INFO] CPU offload enabled")
            except Exception as e:
                if debug:
                    print(f"[WARN] CPU offload not available: {e}")
    else:
        if enable_cpu_offload:
            try:
                pipe.enable_model_cpu_offload()
                if debug:
                    print("[INFO] CPU offload enabled on CPU device")
            except Exception as e:
                if debug:
                    print(f"[WARN] CPU offload not available: {e}")

    return pipe


def resize_with_aspect_fill(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """Resize with aspect fill and center crop to exact dimensions."""
    src_w, src_h = image.size
    target_ratio = target_width / target_height
    src_ratio = src_w / src_h if src_h != 0 else target_ratio

    if src_ratio > target_ratio:
        # Source is wider: match height, crop width
        new_h = target_height
        new_w = int(round(new_h * src_ratio))
    else:
        # Source is taller or equal: match width, crop height
        new_w = target_width
        new_h = int(round(new_w / src_ratio))

    resized = image.resize((new_w, new_h), resample=Image.BICUBIC)
    left = (new_w - target_width) // 2
    top = (new_h - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    return resized.crop((left, top, right, bottom))


def load_and_prepare_image(path: Path, width: int, height: int) -> Image.Image:
    """Load an image and prepare it for the pipeline (RGB, aspect-fill to model size)."""
    image = Image.open(path).convert("RGB")
    return resize_with_aspect_fill(image, width, height)


def normalize_frames(frames_obj) -> List[Image.Image]:
    """Normalize output frames to a flat list of PIL Images."""
    if isinstance(frames_obj, list):
        if len(frames_obj) == 0:
            return []
        if isinstance(frames_obj[0], Image.Image):
            return frames_obj
        if isinstance(frames_obj[0], list):
            inner = frames_obj[0]
            if len(inner) == 0:
                return []
            if isinstance(inner[0], Image.Image):
                return inner
    # Fallback: attempt to convert numpy tensors/arrays to PIL
    if torch.is_tensor(frames_obj):
        arr = (frames_obj.detach().cpu().clamp(0, 1).numpy() * 255).astype("uint8")
        return [Image.fromarray(frame) for frame in arr]
    if isinstance(frames_obj, np.ndarray):
        arr = np.clip(frames_obj, 0, 255).astype("uint8")
        return [Image.fromarray(frame) for frame in arr]
    raise ValueError("Unrecognized frames output format from pipeline")


def generate_segment(
    pipe: StableVideoDiffusionPipeline,
    conditioning_image: Image.Image,
    num_frames: int,
    motion_bucket_id: int,
    noise_aug_strength: float,
    seed: Optional[int],
    debug: bool,
) -> List[Image.Image]:
    """Generate one video segment conditioned on a single image."""
    generator = None
    if seed is not None:
        device = pipe.device if hasattr(pipe, "device") else "cpu"
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    if debug:
        print(f"[INFO] Generating segment: frames={num_frames}, motion_bucket_id={motion_bucket_id}, noise_aug={noise_aug_strength}, seed={seed}")

    result = pipe(
        image=conditioning_image,
        num_frames=num_frames,
        motion_bucket_id=motion_bucket_id,
        noise_aug_strength=noise_aug_strength,
        generator=generator,
    )

    frames = normalize_frames(result.frames)
    if debug:
        print(f"[INFO] Generated {len(frames)} frames")
    return frames


def autoregressive_rollout(
    pipe: StableVideoDiffusionPipeline,
    start_image: Image.Image,
    total_steps: int,
    segment_frames: int,
    overlap: int,
    motion_bucket_id: int,
    noise_aug_strength: float,
    seed: Optional[int],
    debug: bool,
) -> List[Image.Image]:
    """Generate multiple segments by feeding the last frame back as the next conditioning image.

    This simple autoregressive strategy can introduce drift. Adjust noise_aug_strength and motion_bucket_id to control motion.
    """
    if total_steps <= 0:
        return generate_segment(pipe, start_image, segment_frames, motion_bucket_id, noise_aug_strength, seed, debug)

    all_frames: List[Image.Image] = []
    current_image = start_image
    current_seed = seed
    for step_index in range(total_steps + 1):
        segment_seed = None if current_seed is None else current_seed + step_index
        segment = generate_segment(
            pipe=pipe,
            conditioning_image=current_image,
            num_frames=segment_frames,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            seed=segment_seed,
            debug=debug,
        )

        if step_index == 0:
            all_frames.extend(segment)
        else:
            # Overlap handling: drop the first N frames to stitch smoothly
            start_idx = max(0, min(len(segment), overlap))
            all_frames.extend(segment[start_idx:])

        current_image = segment[-1]

    return all_frames


def save_frames_to_directory(frames: List[Image.Image], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        frame_path = output_dir / f"frame_{idx:05d}.png"
        frame.save(frame_path)


def save_mp4(frames: List[Image.Image], output_path: Path, fps: int) -> None:
    if imageio is None:
        raise RuntimeError("imageio not installed. Install with: pip install imageio imageio-ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(output_path), fps=fps, codec="libx264", quality=8) as writer:
        for frame in frames:
            writer.append_data(np.array(frame))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autoregressive image-to-video next-frame generator using a pretrained diffusion pipeline.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to the conditioning input image (RGB)")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory for frames and videos")
    parser.add_argument("--model-id", type=str, default="stabilityai/stable-video-diffusion-img2vid-xt", help="Hugging Face model id for image-to-video pipeline")
    parser.add_argument("--width", type=int, default=1024, help="Model input width (recommend 1024 for XT)")
    parser.add_argument("--height", type=int, default=576, help="Model input height (recommend 576 for XT)")
    parser.add_argument("--num-frames", type=int, default=25, help="Frames per generated segment")
    parser.add_argument("--motion-bucket-id", type=int, default=127, help="Motion strength bucket (higher = more motion)")
    parser.add_argument("--noise-aug-strength", type=float, default=0.02, help="Noise augmentation strength for conditioning image")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--rollout-steps", type=int, default=0, help="Number of additional autoregressive segments to generate")
    parser.add_argument("--overlap", type=int, default=1, help="Number of frames to overlap between segments to avoid duplicates")
    parser.add_argument("--save-mp4", action="store_true", help="Also save an MP4 video of the frames")
    parser.add_argument("--mp4-fps", type=int, default=12, help="FPS for saved MP4 video")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto|cuda|mps|cpu")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 if supported (recommended on CUDA)")
    parser.add_argument("--enable-attention-slicing", action="store_true", help="Enable attention slicing for lower memory usage")
    parser.add_argument("--enable-vae-slicing", action="store_true", help="Enable VAE slicing for lower memory usage")
    parser.add_argument("--enable-cpu-offload", action="store_true", help="Enable model CPU offload to reduce VRAM usage")
    parser.add_argument("--debug", action="store_true", help="Print verbose debug information")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    frames_dir = output_dir / "frames"
    video_path = output_dir / "video.mp4"

    if not input_path.exists():
        print(f"[ERROR] Input image not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    device_str, dtype, use_half, hf_variant = select_device_and_dtype(args.device, args.fp16)
    if args.debug:
        print(f"[INFO] Using device={device_str}, dtype={dtype}, half={use_half}")

    pipe = load_pipeline(
        model_id=args.model_id,
        device_str=device_str,
        dtype=dtype,
        hf_variant=hf_variant,
        enable_attention_slicing=args.enable_attention_slicing,
        enable_vae_slicing=args.enable_vae_slicing,
        enable_cpu_offload=args.enable_cpu_offload,
        debug=args.debug,
    )

    conditioning_image = load_and_prepare_image(input_path, args.width, args.height)

    frames = autoregressive_rollout(
        pipe=pipe,
        start_image=conditioning_image,
        total_steps=args.rollout_steps,
        segment_frames=args.num_frames,
        overlap=args.overlap,
        motion_bucket_id=args.motion_bucket_id,
        noise_aug_strength=args.noise_aug_strength,
        seed=args.seed,
        debug=args.debug,
    )

    if len(frames) == 0:
        print("[ERROR] No frames generated", file=sys.stderr)
        sys.exit(2)

    save_frames_to_directory(frames, frames_dir)
    if args.debug:
        print(f"[INFO] Saved {len(frames)} frames to: {frames_dir}")

    if args.save_mp4:
        try:
            save_mp4(frames, video_path, fps=args.mp4_fps)
            if args.debug:
                print(f"[INFO] Saved MP4 to: {video_path}")
        except Exception as e:
            print(f"[WARN] Failed to save MP4: {e}", file=sys.stderr)

    print("Done.")


if __name__ == "__main__":
    main()