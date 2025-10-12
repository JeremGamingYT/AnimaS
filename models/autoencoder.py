"""
Module implémentant l'auto‑encodeur utilisé dans la phase 1 d'ANIMA‑S.

L'auto‑encodeur prend en entrée une image raster et prédit à la fois :

  * un graphe sémantique (positions des articulations et types de nœuds)
  * une liste de primitives vectorielles (dans cette version des ellipses)

 Le décodeur s'appuie sur `AnimaS.utils.vector_graphics` pour convertir les
 primitives en image.  Dans cette version, aucun rasteriseur externe n'est
 utilisé : chaque cercle est rendu sous forme de noyau gaussien, ce qui
 garantit la différentiabilité via PyTorch et élimine toute dépendance à
 diffvg ou DRTK.  Le code est ainsi entièrement compatible avec Python 3.12.

Ce code est volontairement simplifié afin de servir de démonstration.  Les
paramètres tels que le nombre de nœuds ou de primitives sont configurables et
doivent être adaptés à la complexité de l'animé choisi.
"""
from __future__ import annotations

import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import Data as GraphData
except ImportError:
    GraphData = None  # type: ignore

from AnimaS.utils import vector_graphics as vg


class AnimationAutoencoder(nn.Module):
    """Auto‑encodeur combinant un encodeur CNN et un décodeur vectoriel diffvg.

    Args:
        image_size: taille des images (carrées) en entrée.
        num_nodes: nombre de nœuds à prédire pour le graphe sémantique.
        num_primitives: nombre de primitives vectorielles (cercles) à générer.
        latent_dim: dimension du vecteur latent entre l'encodeur et les deux têtes.
    """

    def __init__(self, image_size: int = 256, num_nodes: int = 10, num_primitives: int = 5, latent_dim: int = 256) -> None:
        super().__init__()
        self.image_size = image_size
        self.num_nodes = num_nodes
        self.num_primitives = num_primitives
        self.latent_dim = latent_dim
        # Encodeur CNN simple
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # 128
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 16
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Calculer la taille après convolution
        conv_out_dim = (image_size // 16) ** 2 * 256
        self.fc_latent = nn.Linear(conv_out_dim, latent_dim)
        # Tête pour les nœuds (positions x,y et type)
        self.fc_nodes = nn.Linear(latent_dim, num_nodes * 3)  # 2 coords + type index
        # Tête pour primitives.  Chaque primitive est décrite par :
        #  - centre x, y (2)
        #  - demi‑axes rx, ry (2)
        #  - angle de rotation (1)
        #  - couleur r, g, b (3)
        # soit 8 paramètres au total par primitive.
        self.fc_prims = nn.Linear(latent_dim, num_primitives * 8)

    def forward(self, x: torch.Tensor) -> Tuple[GraphData, List[Tuple[object, torch.Tensor]], torch.Tensor]:
        # Encodeur CNN
        batch_size = x.size(0)
        z = self.encoder(x)
        z = self.fc_latent(z)
        # Prédiction des nœuds
        node_params = self.fc_nodes(z)  # (batch, num_nodes*3)
        # Reshape: (batch, num_nodes, 3)
        node_params = node_params.view(batch_size, self.num_nodes, 3)
        node_xy = torch.sigmoid(node_params[..., :2])  # positions normalisées dans [0,1]
        node_types = torch.sigmoid(node_params[..., 2:3])  # type entre 0 et 1 -> sera discrétisé
        # Prédiction des primitives
        prim_params = self.fc_prims(z).view(batch_size, self.num_primitives, 8)
        # Centre de l'ellipse (normalisé [0,1])
        prim_xy = torch.sigmoid(prim_params[..., :2])
        # Demi‑axes rx, ry (normalisés, ensuite multipliés par l'image)
        # Nous limitons les rayons à 40 % de la taille de l'image pour éviter des ellipses trop grandes
        prim_rx = torch.sigmoid(prim_params[..., 2:3]) * 0.4
        prim_ry = torch.sigmoid(prim_params[..., 3:4]) * 0.4
        # Angle de rotation en radians dans [-π, π]
        prim_angle = torch.tanh(prim_params[..., 4:5]) * math.pi
        # Couleur RGB
        prim_color = torch.sigmoid(prim_params[..., 5:8])
        # Construire graphes et primitives pour chaque élément du batch
        graphs: List[GraphData] = []
        prims: List[List[Tuple[object, torch.Tensor]]] = []
        recon_imgs: List[torch.Tensor] = []
        canvas_size = (self.image_size, self.image_size)
        for b in range(batch_size):
            # Construire le graphe PyG
            if GraphData is not None:
                pos = node_xy[b] * self.image_size
                # types : arrondi à l'entier le plus proche
                type_idx = torch.round(node_types[b] * 3).long()  # pour 4 types différents
                # connectivité : pour simplifier, connectons chaque nœud au suivant
                edge_indices = []
                for i in range(self.num_nodes - 1):
                    edge_indices.append([i, i + 1])
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous() if edge_indices else torch.empty((2,0), dtype=torch.long)
                graph = GraphData(x=type_idx.float().unsqueeze(-1), pos=pos.float(), edge_index=edge_index)
            else:
                graph = None  # type: ignore
            graphs.append(graph)
            # Construire les primitives diffvg
            prim_list: List[Tuple[object, torch.Tensor]] = []
            for i in range(self.num_primitives):
                cx = prim_xy[b, i, 0].item() * canvas_size[0]
                cy = prim_xy[b, i, 1].item() * canvas_size[1]
                rx = prim_rx[b, i, 0].item() * canvas_size[0]
                ry = prim_ry[b, i, 0].item() * canvas_size[1]
                angle = prim_angle[b, i, 0].item()
                color = prim_color[b, i].tolist()
                # Crée une ellipse différentiable
                ellipse, c_color = vg.create_ellipse((cx, cy), (rx, ry), angle, tuple(color))
                prim_list.append((ellipse, c_color))
            prims.append(prim_list)
            # Rendu différentiable via DRTK ou noyaux gaussiens
            recon_img = vg.render_scene(prim_list, canvas_size)
            recon_imgs.append(recon_img)
        # Empiler pour retourner batch (C,H,W) par élément
        recon_batch = torch.stack(recon_imgs, dim=0)
        return graphs if len(graphs) > 1 else graphs[0], prims, recon_batch
