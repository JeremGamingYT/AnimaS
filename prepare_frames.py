import argparse
from pathlib import Path
from typing import Tuple, Iterable

from PIL import Image


def _get_resample_filter():
    try:
        return Image.Resampling.LANCZOS  # Pillow >= 9.1
    except AttributeError:
        return Image.LANCZOS  # Pillow < 9.1


def resize_and_letterbox(img: Image.Image, size: int, background_color: Tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """Resize image to fit within size×size and pad to square with background_color.

    Keeps aspect ratio (no distortion). Centers the image on a square canvas.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    # Ensure RGBA to preserve transparency while compositing on background
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    src_w, src_h = img.size
    scale = min(size / src_w, size / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    resample = _get_resample_filter()
    resized = img.resize((new_w, new_h), resample=resample)

    canvas = Image.new("RGB", (size, size), background_color)
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    # Composite resized RGBA over RGB canvas using alpha as mask
    canvas.paste(resized, (left, top), mask=resized.split()[-1])
    return canvas


def iter_image_files(root: Path, extensions: Iterable[str]) -> Iterable[Path]:
    exts = {e.lower() if e.startswith('.') else f'.{e.lower()}' for e in extensions}
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def process_directory(src: Path, dst: Path, size: int, extensions: Iterable[str]) -> None:
    files = list(iter_image_files(src, extensions))
    if not files:
        print(f"No images found in {src} with extensions: {', '.join(extensions)}")
        return
    for fp in files:
        rel = fp.relative_to(src)
        out_path = (dst / rel).with_suffix('.png')  # Save all outputs as PNG
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with Image.open(fp) as im:
                out_img = resize_and_letterbox(im, size)
                out_img.save(out_path, format='PNG')
        except Exception as e:
            print(f"Skipping {fp}: {e}")
    print(f"Done. Wrote images to {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resize and center-pad images to a square without distortion.")
    parser.add_argument('--src', required=True, type=str, help='Source directory with original images (e.g., 1920x1080 frames)')
    parser.add_argument('--dst', required=True, type=str, help='Destination directory for resized images')
    parser.add_argument('--size', type=int, default=256, help='Output square size (e.g., 256 or 128)')
    parser.add_argument('--ext', type=str, nargs='*', default=['.png', '.jpg', '.jpeg'], help='Input image extensions to include')
    args = parser.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)

    if not src_dir.exists() or not src_dir.is_dir():
        raise SystemExit(f"Source directory not found: {src_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)

    process_directory(src_dir, dst_dir, args.size, args.ext)


if __name__ == '__main__':
    main()


