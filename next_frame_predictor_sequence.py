import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

# --- 1. Dataset (Peu de changements) ---
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
            # Normalisation pour correspondre à ce que le modèle VGG attend
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.files) - 2

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path1, img_path2, img_path3 = self.files[idx], self.files[idx+1], self.files[idx+2]
        
        image1 = Image.open(img_path1).convert("RGB")
        image2 = Image.open(img_path2).convert("RGB")
        target_image = Image.open(img_path3).convert("RGB")

        tensor1 = self.transform(image1)
        tensor2 = self.transform(image2)
        target = self.transform(target_image)
        
        input_tensor = torch.cat([tensor1, tensor2], dim=0)
        return input_tensor, target

# --- 2. Architecture du Modèle (Votre U-Net amélioré) ---
class GeneratorUNet(nn.Module):
    """Architecture U-Net utilisée comme générateur."""
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
        return out

# --- 3. NOUVEAU : Perte Perceptuelle (VGG Loss) ---
class PerceptualLoss(nn.Module):
    """Calcule la perte basée sur les caractéristiques extraites d'un VGG pré-entraîné."""
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:23].to(device).eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.loss_fn = nn.L1Loss()
        # Normalisation pour les images entrant dans VGG
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, input_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        # Les entrées du dataset sont déjà normalisées, pas besoin de le refaire ici.
        input_features = self.vgg(input_img)
        target_features = self.vgg(target_img)
        return self.loss_fn(input_features, target_features)

# --- 4. Fonctions d'Entraînement et de Génération ---

def train(args: argparse.Namespace, device):
    """Fonction principale pour l'entraînement."""
    dataset = AnimeFrameSequenceDataset(args.data_dir, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    model = GeneratorUNet(in_channels=6, out_channels=3).to(device)
    
    # Deux types de perte : L1 pour la structure globale, Perceptuelle pour les détails
    criterion_l1 = nn.L1Loss()
    criterion_perceptual = PerceptualLoss(device)
    lambda_l1 = 1.0  # Poids pour la perte L1
    lambda_perceptual = 0.2 # Poids pour la perte perceptuelle

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("Début de l'entraînement avec Perte Perceptuelle...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            preds = model(inputs)
            
            # Calcul des deux pertes
            loss_l1 = criterion_l1(preds, targets)
            loss_perceptual = criterion_perceptual(preds, targets)
            
            # Perte combinée
            loss = lambda_l1 * loss_l1 + lambda_perceptual * loss_perceptual
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{args.epochs} - Perte Combinée : {avg_loss:.6f}")

        if (epoch + 1) % args.save_every == 0:
            save_prediction_examples(model, loader, device, epoch + 1, args.out_dir)
            model_path = Path(args.checkpoint_dir) / f"unet_perceptual_epoch_{epoch+1}.pth"
            model_path.parent.mkdir(exist_ok=True, parents=True)
            torch.save(model.state_dict(), str(model_path))
            print(f"Modèle et exemples sauvegardés pour l'époque {epoch+1}")
            
    print("Entraînement terminé.")

def generate_sequence(args: argparse.Namespace, device):
    """Fonction pour générer une séquence à partir d'un modèle entraîné."""
    model = GeneratorUNet().to(device)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    except FileNotFoundError:
        print(f"Erreur: Fichier modèle non trouvé à l'adresse '{args.model_path}'")
        return
    model.eval()
    print("Modèle chargé.")

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        img1 = Image.open(args.start_frame_1).convert("RGB")
        img2_path = args.start_frame_2 if args.start_frame_2 else args.start_frame_1
        img2 = Image.open(img2_path).convert("RGB")
    except FileNotFoundError as e:
        print(f"Erreur: Fichier image non trouvé - {e}")
        return

    tensor1 = transform(img1).to(device)
    tensor2 = transform(img2).to(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    img1.save(output_dir / "frame_000.png")
    img2.save(output_dir / "frame_001.png")

    print(f"Génération de {args.num_frames} frames...")
    with torch.no_grad():
        for i in range(args.num_frames):
            input_tensor = torch.cat([tensor1, tensor2], dim=0).unsqueeze(0)
            predicted_tensor = model(input_tensor).squeeze(0)
            
            save_tensor_as_image(predicted_tensor.cpu(), output_dir / f"frame_{i+2:03d}.png")
            print(f"Frame {i+2} générée.")

            tensor1, tensor2 = tensor2, predicted_tensor
            
    print(f"Séquence sauvegardée dans '{output_dir}'.")

def denormalize(tensor):
    """Dénormalise un tenseur pour l'affichage."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)

def save_tensor_as_image(tensor, path):
    """Sauvegarde un tenseur dénormalisé en tant qu'image."""
    img_tensor = denormalize(tensor)
    img = transforms.ToPILImage()(img_tensor)
    img.save(path)

def save_prediction_examples(model, loader, device, epoch, output_dir):
    """Sauvegarde quelques exemples de prédictions pendant l'entraînement."""
    model.eval()
    inputs, _ = next(iter(loader))
    inputs = inputs.to(device)
    
    with torch.no_grad():
        preds = model(inputs)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(f"Prédictions à l'Époque {epoch}", fontsize=16)

    for i in range(min(3, inputs.size(0))):
        input_frame_1 = denormalize(inputs[i][:3].cpu())
        input_frame_2 = denormalize(inputs[i][3:].cpu())
        predicted_frame = denormalize(preds[i].cpu())
        
        axes[i, 0].imshow(input_frame_1.permute(1, 2, 0))
        axes[i, 0].set_title(f"Frame T-1")
        axes[i, 1].imshow(input_frame_2.permute(1, 2, 0))
        axes[i, 1].set_title(f"Frame T")
        axes[i, 2].imshow(predicted_frame.permute(1, 2, 0))
        axes[i, 2].set_title("Prédiction T+1")
        for ax in axes[i]: ax.axis("off")
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    plt.savefig(Path(output_dir) / f"epoch_{epoch:04d}.png")
    plt.close(fig)
    model.train()

# --- 5. Point d'Entrée Principal ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîner ou générer des séquences de frames avec un U-Net et une perte perceptuelle.")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "generate"], help="Choisir le mode : 'train' ou 'generate'.")
    
    # Arguments pour l'entraînement
    train_args = parser.add_argument_group('Arguments pour l\'entraînement')
    train_args.add_argument("--data_dir", type=str, help="Chemin du dossier contenant les frames pour l'entraînement.")
    train_args.add_argument("--epochs", type=int, default=150, help="Nombre d'époques.")
    train_args.add_argument("--batch_size", type=int, default=4, help="Taille du batch.")
    train_args.add_argument("--lr", type=float, default=1e-4, help="Taux d'apprentissage.")
    train_args.add_argument("--save_every", type=int, default=10, help="Fréquence de sauvegarde des exemples (en époques).")

    # Arguments pour la génération
    gen_args = parser.add_argument_group('Arguments pour la génération')
    gen_args.add_argument("--model_path", type=str, help="Chemin vers le modèle (.pth) pour la génération.")
    gen_args.add_argument("--start_frame_1", type=str, help="Chemin vers la première image de la séquence.")
    gen_args.add_argument("--start_frame_2", type=str, help="(Optionnel) Chemin vers la deuxième image.")
    gen_args.add_argument("--num_frames", type=int, default=10, help="Nombre de frames à générer.")

    # Arguments communs
    parser.add_argument("--image_size", type=int, default=256, help="Taille des images.")
    parser.add_argument("--out_dir", type=str, default="outputs_perceptual", help="Dossier de sortie pour les images/exemples.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_perceptual", help="Dossier de sortie pour les modèles.")
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device : {device}")

    if args.mode == "train":
        if not args.data_dir:
            parser.error("--data_dir est requis pour le mode 'train'.")
        train(args, device)
    elif args.mode == "generate":
        if not args.model_path or not args.start_frame_1:
            parser.error("--model_path et --start_frame_1 sont requis pour le mode 'generate'.")
        generate_sequence(args, device)