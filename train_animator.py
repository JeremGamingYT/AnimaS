"""
Script d'entraînement pour la phase 3 (modèle séquentiel de prédiction de
graphes).

On attend un fichier JSON contenant une liste de séquences de graphes (chemins
vers des fichiers JSON de graphes) pour chaque scène.  Chaque séquence est
utilisée pour entraîner le modèle `GraphSequenceModel` à prédire le graphe
suivant à partir de n précédents.
"""
import argparse
import json
import os
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from AnimaS.data.dataset import AnimationFrameDataset
from AnimaS.models.animator import GraphSequenceModel

try:
    from torch_geometric.data import Data as GraphData
except ImportError:
    GraphData = None  # type: ignore


class GraphSequenceDataset(Dataset):
    """Dataset de séquences de graphes pour la phase 3."""
    def __init__(self, json_path: str, seq_length: int = 3) -> None:
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.sequences = data['sequences']  # liste de listes de chemins
        self.seq_length = seq_length

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq_paths = self.sequences[idx]
        # Charger graphes
        def load_graph(path: str) -> GraphData:
            with open(path, 'r') as f:
                graph_info = json.load(f)
            return AnimationFrameDataset._graph_from_json(graph_info['graph'])
        graphs = [load_graph(p) for p in seq_paths]
        # on découpe en (inputs, target)
        # inputs: premiers seq_length graphes; target: graph suivant
        inputs = graphs[:self.seq_length]
        target = graphs[self.seq_length]
        return inputs, target


def train(args: argparse.Namespace) -> None:
    if GraphData is None:
        raise RuntimeError("PyTorch Geometric doit être installé pour entraîner le modèle séquentiel.")
    dataset = GraphSequenceDataset(args.sequence_data, seq_length=args.seq_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    # Déterminer le nombre de nœuds et dimension d'entrée à partir d'un exemple
    sample_inputs, sample_target = dataset[0]
    num_nodes = sample_inputs[0].num_nodes
    input_dim = sample_inputs[0].x.size(1) + 2  # type + pos
    model = GraphSequenceModel(num_nodes=num_nodes, input_dim=input_dim,
                               hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for inputs, target in loader:
            # inputs est une liste de listes; target est liste GraphData
            optimizer.zero_grad()
            losses = []
            for inp_seq, tgt in zip(inputs, target):
                inp_seq = [g.to(device) for g in inp_seq]
                tgt = tgt.to(device)
                pred = model(inp_seq)
                loss_pos = mse(pred.pos, tgt.pos)
                loss_type = mse(pred.x, tgt.x)
                loss = loss_pos + args.type_weight * loss_type
                loss.backward()
                losses.append(loss.item())
            optimizer.step()
            total_loss += sum(losses)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}")
        if (epoch + 1) % args.save_every == 0:
            os.makedirs(args.out_dir, exist_ok=True)
            save_path = os.path.join(args.out_dir, f"animator_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Modèle séquentiel sauvegardé dans {save_path}")
    # sauvegarde finale
    os.makedirs(args.out_dir, exist_ok=True)
    final_path = os.path.join(args.out_dir, "animator_rnn.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Modèle final sauvegardé dans {final_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ANIMA-S graph sequence model")
    parser.add_argument('--sequence_data', type=str, required=True, help='Fichier JSON contenant les séquences de graphes')
    parser.add_argument('--seq_length', type=int, default=3, help="Nombre de graphes d'entrée pour prédire le suivant")
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension cachée du modèle séquentiel')
    parser.add_argument('--num_layers', type=int, default=2, help='Nombre de couches LSTM')
    parser.add_argument('--batch_size', type=int, default=1, help='Taille de batch')
    parser.add_argument('--epochs', type=int, default=100, help="Nombre d'époques d'entraînement")
    parser.add_argument('--lr', type=float, default=1e-3, help='Taux d\'apprentissage')
    parser.add_argument('--type_weight', type=float, default=1.0, help='Poids pour la perte des types de nœuds')
    parser.add_argument('--save_every', type=int, default=20, help='Sauvegarder le modèle toutes les N époques')
    parser.add_argument('--out_dir', type=str, default='checkpoints', help='Dossier de sortie pour les modèles sauvegardés')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)