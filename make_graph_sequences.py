import argparse
import json
from pathlib import Path
from typing import List


def list_graph_jsons(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob('*.json'):
        files.append(p)
    files.sort()
    return files


def build_sequences(files: List[Path], seq_length: int, stride: int) -> List[List[str]]:
    # Each sequence entry must have length seq_length + 1 (inputs + target)
    win = seq_length + 1
    sequences: List[List[str]] = []
    for i in range(0, len(files) - win + 1, max(1, stride)):
        seq = [str(p) for p in files[i:i + win]]
        sequences.append(seq)
    return sequences


def main() -> None:
    parser = argparse.ArgumentParser(description='Create graph_sequences.json for train_animator.py from a directory of per-frame graph JSONs.')
    parser.add_argument('--graphs_dir', required=True, type=str, help='Directory containing per-frame graph JSON files (from make_graph_pairs export step)')
    parser.add_argument('--seq_length', type=int, default=3, help='Number of input graphs for the model (target will be the next one)')
    parser.add_argument('--stride', type=int, default=1, help='Stride between windows')
    parser.add_argument('--out', required=True, type=str, help='Output path for graph_sequences.json')
    args = parser.parse_args()

    graphs_root = Path(args.graphs_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = list_graph_jsons(graphs_root)
    if len(files) < args.seq_length + 1:
        raise SystemExit('Not enough graphs to build sequences. Provide more frames or reduce --seq_length.')

    sequences = build_sequences(files, args.seq_length, args.stride)
    with open(out_path, 'w') as f:
        json.dump({'sequences': sequences}, f, indent=2)
    print(f'Wrote {len(sequences)} sequences to {out_path}')


if __name__ == '__main__':
    main()