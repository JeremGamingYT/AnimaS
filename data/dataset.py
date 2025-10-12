"""
Dataset utilities for ANIMA‑S.

Ce module fournit des classes permettant de charger des images d'un animé, de les
convertir en représentation interne (graphes sémantiques et primitives
vectorielles) et de les renvoyer sous forme de tenseurs PyTorch.  Les frames
doivent être placées dans un dossier hiérarchisé par scènes et plans
(ex: `scene01_plan03_frame001.png`).

Usage :

>>> from AnimaS.data.dataset import AnimationFrameDataset
>>> dataset = AnimationFrameDataset('/path/to/frames')
>>> image, graph_data, vector_scene = dataset[0]

Le graphe est retourné sous forme d'un objet `torch_geometric.data.Data`
représentant les positions des nœuds et les connexions entre eux.  Les
primitives vectorielles sont représentées par une liste d'objets diffvg (par
exemple `diffvg.Circle`, `diffvg.Path`), ce qui permet de les rendre
différemment dans un autre module.
"""
from __future__ import annotations

import os
import json
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

try:
    import networkx as nx
    from torch_geometric.data import Data as GraphData
except ImportError:
    nx = None  # type: ignore
    GraphData = None  # type: ignore


class AnimationFrameDataset(Dataset):
    """Dataset chargé de gérer des frames d'animation.

    Chaque élément renvoyé comprend :
      * une image raster en format PyTorch (C,H,W) normalisée entre [0,1]
      * un graphe sémantique (positions des articulations, connexions)
      * une liste de primitives vectorielles diffvg

    Les graphes et primitives peuvent être pré‑calculés et stockés dans
    `graph_dir` pour accélérer le chargement.
    """

    def __init__(self, image_dir: str, graph_dir: Optional[str] = None, transform: Optional[transforms.Compose] = None) -> None:
        self.image_dir = image_dir
        self.graph_dir = graph_dir
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
        ])
        # Liste des fichiers image triée pour garantir l'ordre séquentiel
        self.image_files: List[str] = []
        for root, _, files in os.walk(image_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_files.append(os.path.join(root, f))
        self.image_files.sort()

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[GraphData], Optional[List[object]]]:
        # Chargement de l'image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.transform(image)
        # Chargement du graphe pré‑calculé si disponible
        graph_data: Optional[GraphData] = None
        vector_scene: Optional[List[object]] = None
        if self.graph_dir is not None:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            graph_path = os.path.join(self.graph_dir, base_name + '.json')
            if os.path.exists(graph_path):
                with open(graph_path, 'r') as f:
                    graph_info = json.load(f)
                graph_data = self._graph_from_json(graph_info.get('graph'))
                vector_scene = None  # vector primitives can be loaded here if saved
        return image_tensor, graph_data, vector_scene

    @staticmethod
    def _graph_from_json(data: dict) -> Optional[GraphData]:
        """Convertit un graphe sérialisé en objet GraphData.

        Le JSON attendu contient une liste de nœuds avec leurs coordonnées et
        potentiellement des attributs supplémentaires, ainsi qu'une liste d'arêtes.
        """
        if GraphData is None or nx is None:
            # torch_geometric n'est pas installé
            return None
        # Structure attendue : {"nodes": [{"pos": [x,y], "type": "personnage"}, ...],
        # "edges": [[i,j], ...]}
        nodes = data.get('nodes', [])
        edges = data.get('edges', [])
        # Création des attributs
        pos = torch.tensor([n['pos'] for n in nodes], dtype=torch.float)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2,0), dtype=torch.long)
        node_types = [n.get('type', 'unknown') for n in nodes]
        # Convertir les types en entiers via un dictionnaire statique
        type_to_idx = {t: i for i, t in enumerate(sorted(set(node_types)))}
        node_type_idx = torch.tensor([type_to_idx[t] for t in node_types], dtype=torch.long)
        return GraphData(x=node_type_idx.unsqueeze(-1).float(), pos=pos, edge_index=edge_index)

    @staticmethod
    def save_graph(graph: GraphData, primitives: List[object], path: str) -> None:
        """Sérialise un graphe et ses primitives vectorielles vers un fichier JSON."""
        # Conversion du graphe
        nodes = []
        for i, pos in enumerate(graph.pos.cpu().tolist()):
            node_type = int(graph.x[i][0].item()) if graph.x is not None else 0
            nodes.append({'pos': pos, 'type_idx': node_type})
        edges = graph.edge_index.cpu().t().tolist() if graph.edge_index is not None else []
        data = {'graph': {'nodes': nodes, 'edges': edges}, 'primitives': []}
        # Les primitives diffvg ne sont pas directement sérialisables ; l'utilisateur peut 
        # implémenter sa propre sérialisation ici. Pour l'exemple nous sauvegardons
        # uniquement des identifiants symboliques.
        for prim in primitives:
            data['primitives'].append(str(prim))
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
