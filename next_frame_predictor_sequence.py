"""
next_frame_predictor_sequence.py
================================

Version améliorée du script qui entraîne un U-Net à prédire la prochaine
frame (`T+1`) à partir des deux frames précédentes (`T-1` et `T`).

En utilisant une séquence de deux images comme entrée, le modèle peut
apprendre la notion de mouvement (vélocité, direction), ce qui mène à des
prédictions plus logiques et cohérentes.

L'entrée du modèle est un tenseur à 6 canaux (2x images RGB).
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

# --- 1. Chargement des Données (MODIFIÉ pour les séquences) ---

class AnimeFrameSequenceDataset(Dataset):
    """Dataset pour charger les séquences de frames (T-1, T) -> (T+1)."""
    def __init__(self, data_dir: str, image_size: int = 256):
        self.image_size = image_size
        self.files = sorted([p for p in Path(data_dir).glob('*.png')])
        if len(self.files) < 3:
            raise ValueError("Le dossier doit contenir au moins 3 frames.")
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        # MODIFIÉ : On a besoin de 3 frames pour chaque échantillon
        return len(self.files) - 2

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # MODIFIÉ : Charger 3 frames consécutives
        img_path1 = self.files[idx]      # Frame T-1
        img_path2 = self.files[idx + 1]  # Frame T
        img_path3 = self.files[idx + 2]  # Frame T+1 (Cible)
        
        image1 = Image.open(img_path1).convert("RGB")
        image2 = Image.open(img_path2).convert("RGB")
        image3 = Image.open(img_path3).convert("RGB")

        tensor1 = self.transform(image1)
        tensor2 = self.transform(image2)
        target = self.transform(image3)
        
        # MODIFIÉ : Concaténer les deux premières images sur la dimension des canaux
        # L'input est maintenant une image à 6 canaux
        input_tensor = torch.cat([tensor1, tensor2], dim=0)
        
        return input_tensor, target

# --- 2. Architecture du Modèle U-Net (MODIFIÉ pour 6 canaux) ---

class UNet(nn.Module):
    """Architecture U-Net simplifiée pour la prédiction d'images."""
    # MODIFIÉ : Accepte 6 canaux en entrée par défaut
    def __init__(self, in_channels: int = 6, out_channels: int = 3):
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
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
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
        return torch.sigmoid(out)

# --- 3. Boucle d'Entraînement et Visualisation (MODIFIÉ) ---

def save_prediction_examples(model, loader, device, epoch, output_dir="outputs"):
    """Sauvegarde quelques exemples de prédictions."""
    model.eval()
    Path(output_dir).mkdir(exist_ok=True)
    
    inputs, targets = next(iter(loader))
    inputs, targets = inputs.to(device), targets.to(device)
    
    with torch.no_grad():
        preds = model(inputs)

    # MODIFIÉ : Affichage de 4 images pour plus de clarté
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    fig.suptitle(f"Prédictions à l'Époque {epoch}", fontsize=16)

    for i in range(min(3, inputs.size(0))):
        # Séparer l'input de 6 canaux en deux images de 3 canaux
        input_frame_1 = inputs[i][:3].cpu().permute(1, 2, 0)
        input_frame_2 = inputs[i][3:].cpu().permute(1, 2, 0)
        
        axes[i, 0].imshow(input_frame_1)
        axes[i, 0].set_title("Frame T-1")
        
        axes[i, 1].imshow(input_frame_2)
        axes[i, 1].set_title("Frame T")
        
        axes[i, 2].imshow(targets[i].cpu().permute(1, 2, 0))
        axes[i, 2].set_title("Frame T+1 (Réelle)")
        
        axes[i, 3].imshow(preds[i].cpu().permute(1, 2, 0))
        axes[i, 3].set_title("Frame T+1 (Prédite)")
        
        for ax in axes[i]:
            ax.axis("off")
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{output_dir}/epoch_{epoch:04d}.png")
    plt.close(fig)
    

def train(args: argparse.Namespace):
    """Fonction principale pour l'entraînement."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device : {device}")

    dataset = AnimeFrameSequenceDataset(args.data_dir, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # MODIFIÉ : Le modèle prend 6 canaux en entrée
    model = UNet(in_channels=6, out_channels=3).to(device)
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

        if (epoch + 1) % args.save_every == 0:
            save_prediction_examples(model, loader, device, epoch + 1, args.out_dir)
            model_path = Path(args.checkpoint_dir) / f"unet_seq_epoch_{epoch+1}.pth"
            model_path.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), str(model_path))
            print(f"Modèle et exemples sauvegardés pour l'époque {epoch+1}")
            
    print("Entraînement terminé.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîner un U-Net pour la prédiction de séquences de frames d'anime.")
    parser.add_argument("--data_dir", type=str, required=True, help="Chemin du dossier contenant les frames.")
    parser.add_argument("--image_size", type=int, default=256, help="Taille des images (carrées).")
    parser.add_argument("--epochs", type=int, default=100, help="Nombre d'époques.")
    parser.add_argument("--batch_size", type=int, default=4, help="Taille du batch.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Taux d'apprentissage.")
    parser.add_argument("--save_every", type=int, default=10, help="Fréquence de sauvegarde des exemples (en époques).")
    parser.add_argument("--out_dir", type=str, default="outputs_sequence", help="Dossier pour sauvegarder les images d'exemples.")
    parser.add_import argparse
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

# --- 1. Chargement des Données (MODIFIÉ pour les séquences) ---

class AnimeFrameSequenceDataset(Dataset):
    """Dataset pour charger les séquences de frames (T-1, T) -> (T+1)."""
    def __init__(self, data_dir: str, image_size: int = 256):
        self.image_size = image_size
        self.files = sorted([p for p in Path(data_dir).glob('*.png')])
        if len(self.files) < 3:
            raise ValueError("Le dossier doit contenir au moins 3 frames.")
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        # MODIFIÉ : On a besoin de 3 frames pour chaque échantillon
        return len(self.files) - 2

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # MODIFIÉ : Charger 3 frames consécutives
        img_path1 = self.files[idx]      # Frame T-1
        img_path2 = self.files[idx + 1]  # Frame T
        img_path3 = self.files[idx + 2]  # Frame T+1 (Cible)
        
        image1 = Image.open(img_path1).convert("RGB")
        image2 = Image.open(img_path2).convert("RGB")
        image3 = Image.open(img_path3).convert("RGB")

        tensor1 = self.transform(image1)
        tensor2 = self.transform(image2)
        target = self.transform(image3)
        
        # MODIFIÉ : Concaténer les deux premières images sur la dimension des canaux
        # L'input est maintenant une image à 6 canaux
        input_tensor = torch.cat([tensor1, tensor2], dim=0)
        
        return input_tensor, target

# --- 2. Architecture du Modèle U-Net (MODIFIÉ pour 6 canaux) ---

class UNet(nn.Module):
    """Architecture U-Net simplifiée pour la prédiction d'images."""
    # MODIFIÉ : Accepte 6 canaux en entrée par défaut
    def __init__(self, in_channels: int = 6, out_channels: int = 3):
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
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
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
        return torch.sigmoid(out)

# --- 3. Boucle d'Entraînement et Visualisation (MODIFIÉ) ---

def save_prediction_examples(model, loader, device, epoch, output_dir="outputs_sequence"):
    """Sauvegarde quelques exemples de prédictions."""
    model.eval()
    Path(output_dir).mkdir(exist_ok=True)
    
    inputs, targets = next(iter(loader))
    inputs, targets = inputs.to(device), targets.to(device)
    
    with torch.no_grad():
        preds = model(inputs)

    # MODIFIÉ : Affichage de 4 images pour plus de clarté
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    fig.suptitle(f"Prédictions à l'Époque {epoch}", fontsize=16)

    for i in range(min(3, inputs.size(0))):
        # Séparer l'input de 6 canaux en deux images de 3 canaux
        input_frame_1 = inputs[i][:3].cpu().permute(1, 2, 0)
        input_frame_2 = inputs[i][3:].cpu().permute(1, 2, 0)
        
        axes[i, 0].imshow(input_frame_1)
        axes[i, 0].set_title("Frame T-1")
        
        axes[i, 1].imshow(input_frame_2)
        axes[i, 1].set_title("Frame T")
        
        axes[i, 2].imshow(targets[i].cpu().permute(1, 2, 0))
        axes[i, 2].set_title("Frame T+1 (Réelle)")
        
        axes[i, 3].imshow(preds[i].cpu().permute(1, 2, 0))
        axes[i, 3].set_title("Frame T+1 (Prédite)")
        
        for ax in axes[i]:
            ax.axis("off")
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{output_dir}/epoch_{epoch:04d}.png")
    plt.close(fig)
    

def train(args: argparse.Namespace):
    """Fonction principale pour l'entraînement."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device : {device}")

    dataset = AnimeFrameSequenceDataset(args.data_dir, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # MODIFIÉ : Le modèle prend 6 canaux en entrée
    model = UNet(in_channels=6, out_channels=3).to(device)
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

        if (epoch + 1) % args.save_every == 0:
            save_prediction_examples(model, loader, device, epoch + 1, args.out_dir)
            model_path = Path(args.checkpoint_dir) / f"unet_seq_epoch_{epoch+1}.pth"
            model_path.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), str(model_path))
            print(f"Modèle et exemples sauvegardés pour l'époque {epoch+1}")
            
    print("Entraînement terminé.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîner un U-Net pour la prédiction de séquences de frames d'anime.")
    parser.add_argument("--data_dir", type=str, required=True, help="Chemin du dossier contenant les frames.")
    parser.add_argument("--image_size", type=int, default=256, help="Taille des images (carrées).")
    parser.add_argument("--epochs", type=int, default=100, help="Nombre d'époques.")
    parser.add_argument("--batch_size", type=int, default=4, help="Taille du batch.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Taux d'apprentissage.")
    parser.add_argument("--save_every", type=int, default=10, help="Fréquence de sauvegarde des exemples (en époques).")
    parser.add_argument("--out_dir", type=str, default="outputs_sequence", help="Dossier pour sauvegarder les images d'exemples.")
    # LA LIGNE CORRIGÉE EST CI-DESSOUS
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_sequence", help="Dossier pour sauvegarder les modèles.")
    
    args = parser.parse_args()
    train(args)