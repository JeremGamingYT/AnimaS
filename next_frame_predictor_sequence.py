import os
from pathlib import Path
from typing import List, Tuple
import argparse

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
        return len(self.files) - 2

    # MODIFICATION 1 : La fonction retourne maintenant les noms de fichiers
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, tuple[str, str]]:
        img_path1 = self.files[idx]
        img_path2 = self.files[idx + 1]
        img_path3 = self.files[idx + 2]
        
        image1 = Image.open(img_path1).convert("RGB")
        image2 = Image.open(img_path2).convert("RGB")
        image3 = Image.open(img_path3).convert("RGB")

        tensor1 = self.transform(image1)
        tensor2 = self.transform(image2)
        target = self.transform(image3)
        
        input_tensor = torch.cat([tensor1, tensor2], dim=0)
        
        # On retourne aussi les noms de fichiers pour l'affichage
        filenames = (img_path1.name, img_path2.name)
        
        return input_tensor, target, filenames

# --- 2. Architecture du Modèle U-Net (inchangé) ---

class UNet(nn.Module):
    """Architecture U-Net simplifiée pour la prédiction d'images."""
    def __init__(self, in_channels: int = 6, out_channels: int = 3):
        super().__init__()
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = self.conv_block(256, 512)
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

# MODIFICATION 2 : La fonction de sauvegarde utilise les noms de fichiers
def save_prediction_examples(model, loader, device, epoch, output_dir="outputs_sequence"):
    """Sauvegarde quelques exemples de prédictions."""
    model.eval()
    Path(output_dir).mkdir(exist_ok=True)
    
    # On récupère maintenant les 'filenames' en plus des tenseurs
    inputs, targets, filenames = next(iter(loader))
    inputs, targets = inputs.to(device), targets.to(device)
    
    with torch.no_grad():
        preds = model(inputs)

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    fig.suptitle(f"Prédictions à l'Époque {epoch}", fontsize=16)

    for i in range(min(3, inputs.size(0))):
        input_frame_1 = inputs[i][:3].cpu().permute(1, 2, 0)
        input_frame_2 = inputs[i][3:].cpu().permute(1, 2, 0)
        
        # On utilise les noms de fichiers récupérés pour les titres
        axes[i, 0].imshow(input_frame_1)
        axes[i, 0].set_title(f"{filenames[0][i]}")
        
        axes[i, 1].imshow(input_frame_2)
        axes[i, 1].set_title(f"{filenames[1][i]}")
        
        axes[i, 2].imshow(targets[i].cpu().permute(1, 2, 0))
        axes[i, 2].set_title("(Validation)") # Titre fixe comme demandé
        
        axes[i, 3].imshow(preds[i].cpu().permute(1, 2, 0))
        axes[i, 3].set_title("Prédite") # Titre fixe comme demandé
        
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
    
    model = UNet(in_channels=6, out_channels=3).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("Début de l'entraînement...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        # La boucle d'entraînement doit ignorer les noms de fichiers
        for inputs, targets, _ in loader: # On utilise '_' pour ignorer les noms ici
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
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_sequence", help="Dossier pour sauvegarder les modèles.")
    
    args = parser.parse_args()
    train(args)