"""
next_frame_predictor.py
=======================

Un script complet pour entraîner un réseau de neurones U-Net à prédire la
prochaine frame d'une animation à partir de la frame courante. Ce modèle est
conçu pour fonctionner directement avec un dataset d'images 256x256.

L'architecture U-Net est particulièrement efficace pour les tâches de
transformation d'image, car ses "skip connections" permettent de conserver les
détails de l'image d'entrée tout en apprenant les transformations complexes
nécessaires pour prédire la frame suivante.

Ce modèle est non-linéaire et beaucoup plus puissant que les approches basées
sur la PCA, capable d'apprendre des mouvements, des fondus et des effets.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# --- 1. Chargement des Données ---

class AnimeFrameDataset(Dataset):
    """Dataset pour charger les paires de frames (t, t+1)."""
    def __init__(self, data_dir: str, image_size: int = 256):
        self.image_size = image_size
        self.files = sorted([p for p in Path(data_dir).glob('*.png')])
        if len(self.files) < 2:
            raise ValueError("Le dossier doit contenir au moins 2 frames.")
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.files) - 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Charger l'image courante (input) et l'image suivante (cible)
        img_path1 = self.files[idx]
        img_path2 = self.files[idx + 1]
        
        image1 = Image.open(img_path1).convert("RGB")
        image2 = Image.open(img_path2).convert("RGB")

        return self.transform(image1), self.transform(image2)

# --- 2. Architecture du Modèle U-Net ---

class UNet(nn.Module):
    """Architecture U-Net simplifiée pour la prédiction d'images."""
    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()

        # Encodeur (partie descendante)
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)

        # Décodeur (partie ascendante)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256) # 256 (from upconv) + 256 (from enc3 skip)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128) # 128 (from upconv) + 128 (from enc2 skip)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)   # 64 (from upconv) + 64 (from enc1 skip)
        
        # Couche finale
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encodeur
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e3))
        
        # Décodeur avec skip connections
        d3 = self.upconv3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        out = self.final_conv(d1)
        return torch.sigmoid(out) # On s'assure que la sortie est entre 0 et 1

# --- 3. Boucle d'Entraînement ---

def save_prediction_examples(model, loader, device, epoch, output_dir="outputs"):
    """Sauvegarde quelques exemples de prédictions."""
    model.eval()
    Path(output_dir).mkdir(exist_ok=True)
    
    inputs, targets = next(iter(loader))
    inputs, targets = inputs.to(device), targets.to(device)
    
    with torch.no_grad():
        preds = model(inputs)

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i in range(3):
        axes[i, 0].imshow(inputs[i].cpu().permute(1, 2, 0))
        axes[i, 0].set_title("Frame T")
        axes[i, 1].imshow(targets[i].cpu().permute(1, 2, 0))
        axes[i, 1].set_title("Frame T+1 (Réelle)")
        axes[i, 2].imshow(preds[i].cpu().permute(1, 2, 0))
        axes[i, 2].set_title("Frame T+1 (Prédite)")
        for ax in axes[i]:
            ax.axis("off")
            
    plt.tight_layout()
    plt.savefig(f"{output_dir}/epoch_{epoch:04d}.png")
    plt.close(fig)
    

def train(args: argparse.Namespace):
    """Fonction principale pour l'entraînement."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device : {device}")

    dataset = AnimeFrameDataset(args.data_dir, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    model = UNet().to(device)
    # L1Loss est souvent meilleure pour la génération d'images (moins de flou que MSE)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("Début de l'entraînement...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{args.epochs} - Perte : {avg_loss:.6f}")

        # Sauvegarder des exemples et le modèle périodiquement
        if (epoch + 1) % args.save_every == 0:
            save_prediction_examples(model, loader, device, epoch + 1, args.out_dir)
            
            model_path = Path(args.checkpoint_dir) / f"unet_epoch_{epoch+1}.pth"
            model_path.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), str(model_path))
            print(f"Modèle sauvegardé : {model_path}")
            
    print("Entraînement terminé.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîner un U-Net pour la prédiction de frames d'anime.")
    parser.add_argument("--data_dir", type=str, required=True, help="Chemin du dossier contenant les frames.")
    parser.add_argument("--image_size", type=int, default=256, help="Taille des images (carrées).")
    parser.add_argument("--epochs", type=int, default=100, help="Nombre d'époques.")
    parser.add_argument("--batch_size", type=int, default=4, help="Taille du batch.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Taux d'apprentissage.")
    parser.add_argument("--save_every", type=int, default=10, help="Fréquence de sauvegarde des exemples (en époques).")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Dossier pour sauvegarder les images d'exemples.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Dossier pour sauvegarder les modèles.")
    
    args = parser.parse_args()
    train(args)