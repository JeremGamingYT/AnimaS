"""
Script d'entraînement pour la phase 1 (auto‑encodeur vectoriel).

Ce script charge un jeu de données d'images, initialise l'auto‑encodeur
(`AnimationAutoencoder`) et entraîne le modèle pour minimiser l'erreur entre
l'image originale et l'image reconstruite.  Par défaut, la perte est la
fonction MSE, mais il est recommandé de combiner avec SSIM pour mesurer la
perception visuelle.  La fonction SSIM est fournie dans ce fichier.

L'entraînement utilise la bibliothèque diffvg pour le rendu différentiable.  Si
diffvg n'est pas installé, le code s'exécutera mais ne produira pas de rendu
utile.
"""
import argparse
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from AnimaS.data.dataset import AnimationFrameDataset
from AnimaS.models.autoencoder import AnimationAutoencoder


def collate_images_only(batch):
    """Custom collate that keeps only the image tensors.

    The dataset returns (image, graph, primitives). We ignore the latter two
    because they can be None or non-collatable types.
    """
    images = [sample[0] for sample in batch if sample is not None and sample[0] is not None]
    return torch.stack(images, dim=0)


def ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calcul approximatif du Structural Similarity Index (SSIM).

    Cette implémentation simplifiée convient pour la perte d'entraînement mais
    n'est pas optimisée.  Basée sur la formulation traditionnelle de l'index de
    similarité structurelle.
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_x = pred.mean(dim=[2, 3], keepdim=True)
    mu_y = target.mean(dim=[2, 3], keepdim=True)
    sigma_x = pred.var(dim=[2, 3], keepdim=True)
    sigma_y = target.var(dim=[2, 3], keepdim=True)
    sigma_xy = ((pred - mu_x) * (target - mu_y)).mean(dim=[2, 3], keepdim=True)
    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    return 1 - ssim.mean()


def train(args: argparse.Namespace) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = AnimationFrameDataset(args.data_path, graph_dir=None)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_images_only,
    )
    model = AnimationAutoencoder(image_size=args.image_size, num_nodes=args.num_nodes,
                                 num_primitives=args.num_primitives, latent_dim=args.latent_dim)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for images in loader:
            images = images.to(device)
            optimizer.zero_grad()
            _, _, recon = model(images)
            recon = recon.to(device)
            loss_mse = mse(recon, images)
            loss_ssim = ssim_loss(recon, images)
            loss = loss_mse + args.ssim_weight * loss_ssim
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        avg_loss = total_loss / len(loader.dataset)
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}")
        # sauvegarder périodiquement
        if (epoch + 1) % args.save_every == 0:
            os.makedirs(args.out_dir, exist_ok=True)
            save_path = os.path.join(args.out_dir, f"autoencoder_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Modèle sauvegardé dans {save_path}")
    # sauvegarde finale
    os.makedirs(args.out_dir, exist_ok=True)
    final_path = os.path.join(args.out_dir, "autoencoder.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Modèle final sauvegardé dans {final_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ANIMA-S autoencoder")
    parser.add_argument('--data_path', type=str, required=True, help="Chemin du dossier d'images")
    parser.add_argument('--image_size', type=int, default=256, help='Taille des images (carrées)')
    parser.add_argument('--num_nodes', type=int, default=10, help='Nombre de nœuds dans le graphe')
    parser.add_argument('--num_primitives', type=int, default=5, help='Nombre de primitives vectorielles')
    parser.add_argument('--latent_dim', type=int, default=256, help='Dimension latente')
    parser.add_argument('--batch_size', type=int, default=4, help='Taille de batch')
    parser.add_argument('--epochs', type=int, default=50, help="Nombre d'époques d'entraînement")
    parser.add_argument('--lr', type=float, default=1e-4, help='Taux d\'apprentissage')
    parser.add_argument('--ssim_weight', type=float, default=1.0, help='Poids de la perte SSIM')
    parser.add_argument('--save_every', type=int, default=10, help='Sauvegarder le modèle toutes les N époques')
    parser.add_argument('--out_dir', type=str, default='checkpoints', help='Dossier de sortie pour les modèles sauvegardés')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)