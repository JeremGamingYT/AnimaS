"""
Modèle séquentiel pour la phase 3 d'ANIMA‑S.

Ce module implémente un réseau de neurones récurrent (LSTM) qui prend en
entrée une séquence de graphes (codés en vecteurs) et prédit le graphe
suivant.  Chaque graphe est converti en un vecteur de dimension fixe en
aplatissant les positions et les types des nœuds.  Le modèle peut être
remplacé par un Transformer pour des dépendances à plus long terme.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

try:
    from torch_geometric.data import Data as GraphData
except ImportError:
    GraphData = None  # type: ignore


class GraphSequenceModel(nn.Module):
    """Réseau séquentiel (LSTM) pour prédire le prochain graphe d'une séquence.

    Args:
        num_nodes: nombre fixe de nœuds dans chaque graphe.
        input_dim: nombre de caractéristiques par nœud (positions 2D + type 1D = 3).
        hidden_dim: dimension des états cachés du LSTM.
        num_layers: nombre de couches LSTM.
    """

    def __init__(self, num_nodes: int = 10, input_dim: int = 3, hidden_dim: int = 128, num_layers: int = 2) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM prend un vecteur de longueur num_nodes*input_dim à chaque étape
        self.lstm = nn.LSTM(num_nodes * input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, num_nodes * input_dim)

    def forward(self, graphs: List[GraphData]) -> GraphData:
        """Prédit le graphe suivant à partir d'une séquence de graphes.

        Args:
            graphs: liste de GraphData de longueur T.  Chaque graphe doit avoir
                `pos` de forme [num_nodes, 2] et `x` de forme [num_nodes, 1] (type).

        Returns:
            GraphData correspondant à la prédiction du graphe suivant.
        """
        if GraphData is None:
            raise RuntimeError("torch_geometric n'est pas installé. Impossible d'utiliser le modèle séquentiel.")
        T = len(graphs)
        if T == 0:
            raise ValueError("La séquence de graphes ne peut pas être vide.")
        batch_size = 1
        # Construire la séquence des vecteurs
        seq = []
        for g in graphs:
            # Aplatir pos et x
            pos = g.pos.view(self.num_nodes, 2)
            node_type = g.x.view(self.num_nodes, 1)
            vec = torch.cat([pos, node_type], dim=1).view(-1)  # (num_nodes*3)
            seq.append(vec)
        seq_tensor = torch.stack(seq, dim=0).unsqueeze(0)  # (1, T, num_nodes*3)
        lstm_out, _ = self.lstm(seq_tensor)
        last_hidden = lstm_out[:, -1, :]  # (1, hidden_dim)
        next_vec = self.fc_out(last_hidden).view(self.num_nodes, self.input_dim)
        # Séparer positions et type
        pred_pos = next_vec[:, :2]
        pred_type = next_vec[:, 2:3]
        # Conserver la même structure d'arêtes que le dernier graphe
        edge_index = graphs[-1].edge_index
        return GraphData(x=pred_type, pos=pred_pos, edge_index=edge_index)
