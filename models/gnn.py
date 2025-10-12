"""
Implémentation de la phase 2 : réseau de neurones sur graphes (GNN).

Ce module définit un modèle qui prend en entrée un graphe sémantique et un
vecteur de commande (delta) décrivant la modification souhaitée (par exemple
rotation, translation d'une articulation) et renvoie un graphe mis à jour.  Il
utilise PyTorch Geometric pour manipuler les structures de graphes 【501052453445400†L229-L244】.

Pour plus de flexibilité, le modèle est paramétrique : vous pouvez choisir
différents types de convolutions (GCN, GAT, GraphSAGE) et ajuster la profondeur.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv, GraphConv, GATv2Conv
    from torch_geometric.data import Data as GraphData
except ImportError:
    # PyTorch Geometric n'est pas installé; on définit des stubs pour éviter l'échec d'import.
    GCNConv = GraphConv = GATv2Conv = object  # type: ignore
    GraphData = None  # type: ignore


class GraphCorrectionModel(nn.Module):
    """Réseau de neurones sur graphes pour appliquer des modifications locales.

    Args:
        node_feat_dim: dimension d'entrée des caractéristiques des nœuds (par défaut 1 pour le type d'articulation).
        delta_dim: dimension du vecteur de commande (delta) à concaténer à chaque nœud.
        hidden_dim: dimension des couches internes.
        num_layers: nombre de couches de convolution sur graphes.
        conv_type: type de convolution ('gcn', 'graph', 'gat').
    """

    def __init__(self, node_feat_dim: int = 1, delta_dim: int = 4, hidden_dim: int = 64, num_layers: int = 3, conv_type: str = 'gcn') -> None:
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.delta_dim = delta_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        conv_cls = {'gcn': GCNConv, 'graph': GraphConv, 'gat': GATv2Conv}[conv_type]
        # Première couche combine caractéristiques et delta
        self.convs = nn.ModuleList()
        self.convs.append(conv_cls(node_feat_dim + delta_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(conv_cls(hidden_dim, hidden_dim))
        # Dernière couche prédit le déplacement (delta_pos) et le changement de type
        self.conv_out = conv_cls(hidden_dim, 3)  # 2 dims pour dx,dy et 1 dim pour d_type

    def forward(self, data: GraphData, delta: torch.Tensor) -> GraphData:
        """Applique le GNN pour corriger un graphe.

        Args:
            data: graphe d'entrée (positions et types).  On suppose `data.x` de
              forme [num_nodes, node_feat_dim] et `data.pos` de forme [num_nodes, 2].
            delta: vecteur de commande de forme [delta_dim], broadcasté sur les nœuds.

        Returns:
            Graphe mis à jour avec nouvelles positions et types.
        """
        if GraphData is None:
            raise RuntimeError("torch_geometric n'est pas installé. Impossible d'utiliser le modèle GNN.")
        x = data.x
        pos = data.pos
        num_nodes = x.size(0)
        # concaténer delta à chaque nœud
        delta_expand = delta.view(1, -1).repeat(num_nodes, 1)
        h = torch.cat([x, delta_expand], dim=1)
        # propagation à travers les couches
        edge_index = data.edge_index
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
        # sortie : dx, dy, d_type
        out = self.conv_out(h, edge_index)
        delta_pos = out[:, :2]
        delta_type = out[:, 2:3]
        # Mettre à jour positions et types
        new_pos = pos + delta_pos
        new_type = x + delta_type
        return GraphData(x=new_type, pos=new_pos, edge_index=edge_index)
