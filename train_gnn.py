"""
Script d'entraînement pour la phase 2 (GNN de modification).

Le jeu de données attendu est un fichier JSON contenant des paires de graphes
consecutifs avec le vecteur de commande (delta) associé.  Chaque entrée doit
avoir la structure :

```
{
  "pairs": [
    {
      "graph_A": "path/to/graphA.json",
      "graph_B": "path/to/graphB.json",
      "delta": [dx, dy, angle, scale]
    },
    ...
  ]
}
```

Le modèle `GraphCorrectionModel` apprend à prédire `graph_B` à partir de
`graph_A` et de `delta`.  Les positions sont mises à jour et les types de
nœuds ajustés.  La perte est la MSE entre les positions prédites et
celles du graphe cible.
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
from AnimaS.models.gnn import GraphCorrectionModel

try:
    from torch_geometric.data import Data as GraphData
except ImportError:
    GraphData = None  # type: ignore


class GraphPairDataset(Dataset):
    """Dataset pour les paires de graphes et deltas."""
    def __init__(self, json_path: str) -> None:
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.pairs = data['pairs']

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        item = self.pairs[idx]
        # Charger graph_A et graph_B via AnimationFrameDataset helper
        # Chaque json contient un graphe sérialisé
        def load_graph(path: str) -> GraphData:
            with open(path, 'r') as f:
                graph_info = json.load(f)
            return AnimationFrameDataset._graph_from_json(graph_info['graph'])
        graph_A = load_graph(item['graph_A'])
        graph_B = load_graph(item['graph_B'])
        delta = torch.tensor(item['delta'], dtype=torch.float)
        return graph_A, graph_B, delta


def train(args: argparse.Namespace) -> None:
    if GraphData is None:
        raise RuntimeError("PyTorch Geometric doit être installé pour entraîner le GNN.")
    dataset = GraphPairDataset(args.graph_pairs)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    # Déterminer dimension d'entrée et delta
    sample_A, _, sample_delta = dataset[0]
    node_feat_dim = sample_A.x.size(1)
    delta_dim = sample_delta.size(0)
    model = GraphCorrectionModel(node_feat_dim=node_feat_dim, delta_dim=delta_dim,
                                 hidden_dim=args.hidden_dim, num_layers=args.num_layers, conv_type=args.conv_type)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for graph_A, graph_B, delta in loader:
            # graph_A et graph_B sont des listes de GraphData
            # Pour chaque élément du batch, nous devons déplacer les tenseurs sur l'appareil
            optimizer.zero_grad()
            losses = []
            for ga, gb, d in zip(graph_A, graph_B, delta):
                ga = ga.to(device)
                gb = gb.to(device)
                d = d.to(device)
                pred = model(ga, d)
                # calculer la perte sur positions et types
                loss_pos = mse(pred.pos, gb.pos)
                loss_type = mse(pred.x, gb.x)
                loss = loss_pos + args.type_weight * loss_type
                loss.backward()
                losses.append(loss.item())
            optimizer.step()
            total_loss += sum(losses)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}")
        if (epoch + 1) % args.save_every == 0:
            os.makedirs(args.out_dir, exist_ok=True)
            save_path = os.path.join(args.out_dir, f"gnn_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Modèle GNN sauvegardé dans {save_path}")
    # sauvegarde finale
    os.makedirs(args.out_dir, exist_ok=True)
    final_path = os.path.join(args.out_dir, "gnn_corrector.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Modèle final sauvegardé dans {final_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ANIMA-S GNN for graph correction")
    parser.add_argument('--graph_pairs', type=str, required=True, help='Fichier JSON contenant les paires de graphes et les deltas')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Dimension cachée du GNN')
    parser.add_argument('--num_layers', type=int, default=3, help='Nombre de couches GNN')
    parser.add_argument('--conv_type', type=str, default='gcn', choices=['gcn', 'graph', 'gat'], help='Type de convolution')
    parser.add_argument('--batch_size', type=int, default=1, help='Taille de batch (1 par défaut car chaque graphe peut avoir un nombre variable de nœuds)')
    parser.add_argument('--epochs', type=int, default=100, help="Nombre d'époques d'entraînement")
    parser.add_argument('--lr', type=float, default=1e-3, help='Taux d\'apprentissage')
    parser.add_argument('--type_weight', type=float, default=1.0, help='Poids pour la perte des types de nœuds')
    parser.add_argument('--save_every', type=int, default=20, help='Sauvegarder le modèle toutes les N époques')
    parser.add_argument('--out_dir', type=str, default='checkpoints', help='Dossier de sortie pour les modèles sauvegardés')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)