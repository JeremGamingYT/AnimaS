import os
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# --- 1. Chargement des Données (Identique à votre code) ---
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
            # Normalisation importante pour les GANs pour stabiliser l'entraînement
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.files) - 2

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path1 = self.files[idx]
        img_path2 = self.files[idx + 1]
        img_path3 = self.files[idx + 2]
        
        image1 = Image.open(img_path1).convert("RGB")
        image2 = Image.open(img_path2).convert("RGB")
        target_image = Image.open(img_path3).convert("RGB")

        tensor1 = self.transform(image1)
        tensor2 = self.transform(image2)
        target = self.transform(target_image)
        
        input_tensor = torch.cat([tensor1, tensor2], dim=0)
        
        return input_tensor, target

# --- 2. Architecture du Générateur (Votre U-Net) ---
class GeneratorUNet(nn.Module):
    """Votre architecture U-Net, maintenant utilisée comme générateur."""
    def __init__(self, in_channels: int = 6, out_channels: int = 3):
        super().__init__()
        # Le code de votre U-Net est bon, on le garde tel quel.
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
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
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
        # Tanh est souvent utilisée comme fonction d'activation finale dans les GANs
        return torch.tanh(out)

# --- 3. NOUVEAU : Architecture du Discriminateur ---
class Discriminator(nn.Module):
    """Discriminateur PatchGAN. Il détermine si chaque "patch" de l'image est réel ou faux."""
    def __init__(self, in_channels: int = 3): # Il regarde une seule image (3 canaux)
        super().__init__()
        # L'entrée du discriminateur est la concaténation des frames d'entrée et de la frame de sortie (réelle ou générée)
        # Donc in_channels = 6 (input) + 3 (output) = 9
        
        def discriminator_block(in_c, out_c, stride=2, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels + 6, 64, normalize=False), # Entrée: Input (6) + Output (3) = 9 canaux
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512, stride=1),
            nn.Conv2d(512, 1, kernel_size=4, padding=1) # Couche finale pour produire une sortie 1 canal
        )

    def forward(self, img_in: torch.Tensor, img_out: torch.Tensor) -> torch.Tensor:
        # Concaténer l'image d'entrée et l'image de sortie (la "condition")
        x = torch.cat([img_in, img_out], dim=1)
        return self.model(x)

# --- 4. Boucle d'Entraînement Améliorée ---

def train_gan(args: argparse.Namespace):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device : {device}")

    # Initialisation des modèles
    generator = GeneratorUNet().to(device)
    discriminator = Discriminator().to(device)

    # Fonctions de perte
    criterion_gan = nn.BCEWithLogitsLoss() # Perte adversariale
    criterion_pixel = nn.L1Loss() # Perte L1 pour la structure
    lambda_pixel = 100 # Poids pour la perte L1, valeur commune

    # Optimiseurs
    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Données
    dataset = AnimeFrameSequenceDataset(args.data_dir, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    print("Début de l'entraînement GAN...")
    for epoch in range(args.epochs):
        for i, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Labels pour le réel et le faux
            real_label = torch.ones(inputs.size(0), 1, 30, 30, device=device) # La taille dépend de la sortie du Discriminateur
            fake_label = torch.zeros(inputs.size(0), 1, 30, 30, device=device)
            
            # --- Entraînement du Générateur ---
            optimizer_g.zero_grad()
            
            fake_outputs = generator(inputs)
            pred_fake = discriminator(inputs, fake_outputs)
            loss_g_gan = criterion_gan(pred_fake, real_label) # Le générateur veut que le discriminateur pense que ses images sont réelles
            loss_g_pixel = criterion_pixel(fake_outputs, targets)
            
            # Perte totale du générateur
            loss_g = loss_g_gan + lambda_pixel * loss_g_pixel
            loss_g.backward()
            optimizer_g.step()

            # --- Entraînement du Discriminateur ---
            optimizer_d.zero_grad()
            
            # Perte avec les images réelles
            pred_real = discriminator(inputs, targets)
            loss_d_real = criterion_gan(pred_real, real_label)
            
            # Perte avec les images fausses (générées)
            pred_fake = discriminator(inputs, fake_outputs.detach())
            loss_d_fake = criterion_gan(pred_fake, fake_label)
            
            # Perte totale du discriminateur
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            optimizer_d.step()
            
            if i % 100 == 0:
                print(f"[Epoch {epoch+1}/{args.epochs}] [Batch {i}/{len(loader)}] "
                      f"[D loss: {loss_d.item():.4f}] [G loss: {loss_g.item():.4f}]")

        # Sauvegarde des modèles et des exemples
        if (epoch + 1) % args.save_every == 0:
            save_gan_examples(generator, loader, device, epoch + 1, args.out_dir)
            Path(args.checkpoint_dir).mkdir(exist_ok=True)
            torch.save(generator.state_dict(), f"{args.checkpoint_dir}/generator_epoch_{epoch+1}.pth")
            torch.save(discriminator.state_dict(), f"{args.checkpoint_dir}/discriminator_epoch_{epoch+1}.pth")
            print(f"Modèles et exemples sauvegardés pour l'époque {epoch+1}")

def save_gan_examples(generator, loader, device, epoch, output_dir):
    """Sauvegarde des exemples de prédictions du GAN."""
    generator.eval()
    inputs, targets = next(iter(loader))
    inputs, targets = inputs.to(device), targets.to(device)

    with torch.no_grad():
        fake_outputs = generator(inputs)

    # Dénormaliser les images pour l'affichage
    def denorm(img_tensor):
        return img_tensor * 0.5 + 0.5

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(f"Prédictions GAN à l'Époque {epoch}", fontsize=16)

    for i in range(min(3, inputs.size(0))):
        input_frame_1 = denorm(inputs[i][:3].cpu()).permute(1, 2, 0)
        input_frame_2 = denorm(inputs[i][3:].cpu()).permute(1, 2, 0)
        predicted_frame = denorm(fake_outputs[i].cpu()).permute(1, 2, 0)
        
        axes[i, 0].imshow(input_frame_1)
        axes[i, 0].set_title(f"Frame T-1")
        axes[i, 1].imshow(input_frame_2)
        axes[i, 1].set_title(f"Frame T")
        axes[i, 2].imshow(predicted_frame)
        axes[i, 2].set_title("Prédiction T+1")

        for ax in axes[i]: ax.axis("off")
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    Path(output_dir).mkdir(exist_ok=True)
    plt.savefig(f"{output_dir}/epoch_{epoch:04d}.png")
    plt.close(fig)
    generator.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîner un GAN pour la prédiction de séquences de frames.")
    parser.add_argument("--data_dir", type=str, required=True, help="Chemin du dossier contenant les frames.")
    parser.add_argument("--image_size", type=int, default=256, help="Taille des images.")
    parser.add_argument("--epochs", type=int, default=200, help="Nombre d'époques (les GANs nécessitent plus d'entraînement).")
    parser.add_argument("--batch_size", type=int, default=1, help="Taille du batch (1 est commun pour les pix2pix).")
    parser.add_argument("--lr", type=float, default=2e-4, help="Taux d'apprentissage.")
    parser.add_argument("--save_every", type=int, default=10, help="Fréquence de sauvegarde.")
    parser.add_argument("--out_dir", type=str, default="outputs_gan", help="Dossier pour sauvegarder les images d'exemples.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_gan", help="Dossier pour sauvegarder les modèles.")
    
    args = parser.parse_args()
    train_gan(args)