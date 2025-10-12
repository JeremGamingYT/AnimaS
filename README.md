# Projet ANIMA‑S

ANIMA‑S est un projet de recherche visant à créer une intelligence artificielle capable de générer des animations japonaises complètes de manière autonome.  L’objectif est d’encoder la structure sémantique et le style artistique d’un animé existant afin de pouvoir reconstruire et manipuler les images et enfin générer des séquences animées fluides.  Ce dépôt fournit un squelette de code complet répondant aux trois phases décrites dans le cahier des charges : reconstruction fidèle (auto‑encodeur vectoriel), modification contrôlée via un réseau de neurones sur graphes (GNN) et prédiction de séquences (RNN/Transformer).

## Organisation du dépôt

```
anima_s/
├── data/
│   └── dataset.py          # Classes de chargement et pré‑traitement des images et graphes
├── models/
│   ├── autoencoder.py       # Auto‑encodeur CNN → représentation vectorielle → rendu différentiable
│   ├── gnn.py               # Réseau de neurones sur graphes pour appliquer des modifications
│   └── animator.py          # Modèle séquentiel (RNN/Transformer) pour prédire les graphes suivants
├── utils/
│   └── vector_graphics.py   # Outils pour manipuler et rendre des primitives vectorielles sans dépendances externes
├── train_autoencoder.py     # Script d'entraînement pour la phase 1
├── train_gnn.py             # Script d'entraînement pour la phase 2
├── train_animator.py        # Script d'entraînement pour la phase 3
└── main.py                  # Exemple d'utilisation combinée des trois modèles
```

## Dépendances

Le projet est conçu pour **Python 3.10 ou supérieur** et s’appuie sur **PyTorch ≥ 2.2** comme moteur d’apprentissage principal.  Il ne nécessite aucune bibliothèque de rasterisation externe (telle que diffvg ou DRTK), car le rendu différentiable des primitives est implémenté directement dans `utils/vector_graphics.py` à l’aide de noyaux gaussiens.

Les dépendances essentielles sont :

- `torchvision` et `Pillow` : pour le chargement et la transformation d’images.
- `pytorch_geometric` (facultatif) : pour la construction et l’entraînement des réseaux de neurones sur graphes.
- `networkx` : pour la représentation et la manipulation des graphes sémantiques.
- `tqdm` : pour l’affichage de barres de progression.

> **Remarque :** l’entraînement complet bénéficiera de l’utilisation d’un GPU (par exemple un NVIDIA A100 ou H100), mais le code est entièrement compatible avec un CPU.

## Phase 1 : Auto‑encodeur vectoriel

L’objectif de la phase 1 est d’apprendre à reconstruire fidèlement chaque frame de l’animé en la convertissant en une représentation interne structurée (graphe sémantique + primitives vectorielles) et en la re‑rendant au format raster.  L’encodeur est un réseau de neurones convolutionnel qui prédit :

1. **Un graphe sémantique** : positions 2D des articulations, visibilité des segments, identifiants de personnages, etc.  Ce graphe est implémenté avec `networkx` et converti en `torch_geometric.data.Data` pour l’apprentissage.
2. **Un ensemble de primitives vectorielles** : chaque élément de l’image est approximé par une **ellipse** décrite par un centre `(cx, cy)`, deux demi‑axes `(rx, ry)` et un angle d’orientation.  Ces ellipses sont rendues par un moteur interne qui projette chaque forme en une gaussienne anisotrope sur la grille de pixels.  Cette technique s’inspire des méthodes de « Gaussian splatting » tout en permettant des formes allongées et orientées, mieux adaptées aux objets complexes.

Le décodeur prend ces primitives et les convertit en image raster via ce rendu gaussien anisotrope.  La boucle d’entraînement minimise l’erreur structurale (SSIM + L1) entre l’image originale et l’image reconstruite et sauvegarde les graphes ainsi appris.  Les paramètres des ellipses (centre, demi‑axes, angle, couleur) sont optimisés par rétro‑propagation.

## Phase 2 : Réseau de neurones sur graphes (GNN)

Une fois chaque frame représentée par un graphe, la phase 2 apprend comment appliquer des modifications locales (par exemple, lever le bras) de manière cohérente.  On crée un jeu de données de paires `(graph_A, graph_B)` où les graphes représentent des images consécutives ; le **delta** entre les deux encode le mouvement simple.  Un modèle GNN apprend à prédire `graph_B` à partir de `graph_A` et du delta.  Ce type de modèles, dérivés des réseaux convolutionnels, opèrent directement sur des structures de graphe et peuvent traiter des tâches telles que la prédiction de nœuds ou de liens 【501052453445400†L229-L244】.  Dans ce projet, le GNN est implémenté avec `pytorch_geometric`.

## Phase 3 : Modèle séquentiel

La dernière phase consiste à prédire le graphe suivant d’une séquence en utilisant un modèle séquentiel (par exemple un LSTM ou Transformer).  On transforme chaque scène de l’animé en une séquence de graphes `[G1, G2, …, Gn]` et on entraîne le modèle à prédire `Gn+1` à partir des `n` graphes précédents.  Ces modèles s’inspirent des réseaux récurrents utilisés en NLP et vision pour capturer les dépendances temporelles et spatiales.

## Utilisation

1. **Préparation des données :** Placez les frames PNG dans un dossier et exécutez le script de dataset (voir `data/dataset.py`) pour créer les graphes et primitives.
2. **Entraîner l’auto‑encodeur :**

   ```bash
   python train_autoencoder.py --data_path /path/to/frames --epochs 50 --batch_size 4
   ```

3. **Entraîner le GNN :** après avoir généré les graphes, exécutez :

   ```bash
   python train_gnn.py --graph_pairs /path/to/graph_pairs.json --epochs 100
   ```

4. **Entraîner le modèle séquentiel :**

   ```bash
   python train_animator.py --sequence_data /path/to/graph_sequences.json --epochs 100
   ```

5. **Générer une animation :** utilisez `main.py` pour charger un nombre de frames de départ, appliquer des modifications et générer des images :

   ```bash
   python main.py --start_frames frame01.png frame02.png --n_frames 24 --output animation
   ```

Ce dépôt offre des exemples de code pour chaque étape, mais l’entraînement complet nécessitera des ressources matérielles considérables.
