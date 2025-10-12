import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torchvision import transforms

try:
    # Prefer the package name used by your training script
    from AnimaS.data.dataset import AnimationFrameDataset
    from AnimaS.models.autoencoder import AnimationAutoencoder
except ImportError:
    # Fallback if imports use lowercase/underscore elsewhere
    from AnimaS.data.dataset import AnimationFrameDataset  # type: ignore
    from AnimaS.models.autoencoder import AnimationAutoencoder  # type: ignore


def list_images(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob('*'):
        if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
            files.append(p)
    files.sort()
    return files


def load_autoencoder(ckpt_path: Path, image_size: int, num_nodes: int, num_primitives: int, latent_dim: int, device: torch.device) -> AnimationAutoencoder:
    model = AnimationAutoencoder(image_size=image_size, num_nodes=num_nodes, num_primitives=num_primitives, latent_dim=latent_dim)
    state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model


def save_graph_json(graph, out_path: Path) -> None:
    # Graph saving uses the dataset helper for consistent schema
    # When primitives are not used, pass an empty list
    AnimationFrameDataset.save_graph(graph, [], str(out_path))


def compute_delta(graph_a, graph_b) -> Tuple[float, float, float, float]:
    """Compute [dx, dy, angle, scale] between two graphs using node positions.

    - dx, dy: shift of centroid from A to B
    - angle: rotation angle (radians) estimated via 2D Procrustes (SVD)
    - scale: isotropic scale factor from A to B
    """
    pos_a = graph_a.pos.detach().cpu()  # [N,2]
    pos_b = graph_b.pos.detach().cpu()
    mean_a = pos_a.mean(dim=0)
    mean_b = pos_b.mean(dim=0)
    dx_dy = (mean_b - mean_a).tolist()
    a0 = pos_a - mean_a
    b0 = pos_b - mean_b
    # 2x2 covariance
    H = a0.t().mm(b0) / max(1, pos_a.shape[0])
    U, S, Vh = torch.linalg.svd(H)
    R = Vh.t().mm(U.t())
    if torch.linalg.det(R) < 0:
        Vh[-1, :] *= -1
        R = Vh.t().mm(U.t())
    angle = float(torch.atan2(R[1, 0], R[0, 0]))
    denom = float((a0 * a0).sum())
    scale = float(S.sum().item() / denom) if denom > 1e-9 else 1.0
    return float(dx_dy[0]), float(dx_dy[1]), angle, scale


def main() -> None:
    parser = argparse.ArgumentParser(description="Export per-frame graphs with a trained autoencoder and build graph_pairs.json for the GNN training.")
    parser.add_argument('--images_dir', required=True, type=str, help='Directory containing preprocessed frames (e.g., 256x256)')
    parser.add_argument('--autoencoder_ckpt', required=True, type=str, help='Path to autoencoder checkpoint (.pth)')
    parser.add_argument('--image_size', type=int, default=256, help='Model image size used during training')
    parser.add_argument('--num_nodes', type=int, default=10, help='Number of nodes predicted by the autoencoder')
    parser.add_argument('--num_primitives', type=int, default=5, help='Number of vector primitives predicted')
    parser.add_argument('--latent_dim', type=int, default=256, help='Latent dimension of the autoencoder')
    parser.add_argument('--out_graph_dir', required=True, type=str, help='Directory to write per-frame graph JSONs')
    parser.add_argument('--out_pairs', required=True, type=str, help='Output path for graph_pairs.json')
    parser.add_argument('--stride', type=int, default=1, help='Use every k-th frame to form pairs (>=1)')
    args = parser.parse_args()

    images_root = Path(args.images_dir)
    out_graph_dir = Path(args.out_graph_dir)
    out_pairs_path = Path(args.out_pairs)
    out_graph_dir.mkdir(parents=True, exist_ok=True)
    out_pairs_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_autoencoder(Path(args.autoencoder_ckpt), args.image_size, args.num_nodes, args.num_primitives, args.latent_dim, device)

    files = list_images(images_root)
    if len(files) < 2:
        raise SystemExit("Need at least 2 frames to build pairs.")

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    graphs_out_paths: List[Path] = []
    with torch.no_grad():
        for idx, img_path in enumerate(files):
            img = Image.open(img_path).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(device)
            graph, _, _ = model(tensor)
            # The model returns a single GraphData when batch_size==1
            # Move tensors to CPU for serialization
            if hasattr(graph, 'to'):
                graph = graph.to('cpu')
            # Save JSON graph alongside a matching name
            out_path = out_graph_dir / (img_path.stem + '.json')
            save_graph_json(graph, out_path)
            graphs_out_paths.append(out_path)

    # Build pairs with stride
    pairs = []
    step = max(1, int(args.stride))
    for i in range(0, len(graphs_out_paths) - step, step):
        path_a = graphs_out_paths[i]
        path_b = graphs_out_paths[i + step]
        # Load back graphs to compute delta via the dataset helper
        with open(path_a, 'r') as fa:
            ja = json.load(fa)
        with open(path_b, 'r') as fb:
            jb = json.load(fb)
        ga = AnimationFrameDataset._graph_from_json(ja['graph'])
        gb = AnimationFrameDataset._graph_from_json(jb['graph'])
        if ga is None or gb is None:
            raise SystemExit("torch_geometric is required to build graph pairs. Please install PyTorch Geometric.")
        dx, dy, angle, scale = compute_delta(ga, gb)
        pairs.append({
            'graph_A': str(path_a),
            'graph_B': str(path_b),
            'delta': [dx, dy, angle, scale],
        })

    with open(out_pairs_path, 'w') as f:
        json.dump({'pairs': pairs}, f, indent=2)
    print(f"Wrote {len(pairs)} pairs to {out_pairs_path}")


if __name__ == '__main__':
    main()