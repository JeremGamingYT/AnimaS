import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from math import exp

# --- 1. Dataset (Votre code, inchangé) ---
class AnimeFrameSequenceDataset(Dataset):
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

# --- 2. Architecture U-Net (Votre code, inchangé) ---
class UNet(nn.Module):
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

# --- 3. AMÉLIORATION : Fonction de Perte SSIM pour la Qualité d'Image ---
# Cette section est le seul ajout majeur. Elle aide le modèle à créer des images nettes.
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

class SSIMLoss(nn.Module):
    def forward(self, img1, img2):
        return 1.0 - ssim(img1, img2)

# --- 4. Fonctions d'Entraînement et de Génération ---

def train(args: argparse.Namespace, device):
    dataset = AnimeFrameSequenceDataset(args.data_dir, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    model = UNet(in_channels=6, out_channels=3).to(device)
    
    # MODIFICATION : On utilise maintenant une combinaison de deux pertes
    criterion_l1 = nn.L1Loss()
    criterion_ssim = SSIMLoss().to(device)
    alpha = 0.85 # Poids important pour la qualité visuelle (SSIM)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("Début de l'entraînement avec Perte L1 + SSIM pour une meilleure qualité...")
    for epoch in range(args.epochs):
        model.train()
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(inputs)
            
            # Calcul de la perte combinée pour des images plus nettes
            loss_l1 = criterion_l1(preds, targets)
            loss_ssim = criterion_ssim(preds, targets)
            loss = alpha * loss_ssim + (1 - alpha) * loss_l1
            
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{args.epochs} - Perte: {loss.item():.6f}")

        if (epoch + 1) % args.save_every == 0:
            save_prediction_examples(model, loader, device, epoch + 1, args.out_dir)
            model_path = Path(args.checkpoint_dir) / f"unet_final_epoch_{epoch+1}.pth"
            model_path.parent.mkdir(exist_ok=True, parents=True)
            torch.save(model.state_dict(), str(model_path))
    print("Entraînement terminé.")

def generate_sequence(args: argparse.Namespace, device):
    model = UNet().to(device)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    except FileNotFoundError:
        print(f"Erreur: Fichier modèle '{args.model_path}' non trouvé.")
        return
    model.eval()
    print("Modèle chargé.")

    transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor()])
    
    try:
        img1 = Image.open(args.start_frame_1).convert("RGB")
        img2_path = args.start_frame_2 if args.start_frame_2 else args.start_frame_1
        img2 = Image.open(img2_path).convert("RGB")
    except FileNotFoundError as e:
        print(f"Erreur: Fichier image non trouvé - {e}")
        return

    tensor1, tensor2 = transform(img1).to(device), transform(img2).to(device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    img1.save(output_dir / "frame_000.png")
    img2.save(output_dir / "frame_001.png")

    print(f"Génération de {args.num_frames} frames...")
    with torch.no_grad():
        for i in range(args.num_frames):
            input_tensor = torch.cat([tensor1, tensor2], dim=0).unsqueeze(0)
            predicted_tensor = model(input_tensor).squeeze(0)
            
            predicted_img = transforms.ToPILImage()(predicted_tensor.cpu())
            predicted_img.save(output_dir / f"frame_{i+2:03d}.png")
            
            tensor1, tensor2 = tensor2, predicted_tensor
    print(f"Séquence sauvegardée dans '{output_dir}'.")

def save_prediction_examples(model, loader, device, epoch, output_dir):
    model.eval()
    inputs, _ = next(iter(loader))
    inputs = inputs.to(device)
    with torch.no_grad(): preds = model(inputs)
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(f"Prédictions à l'Époque {epoch}", fontsize=16)
    for i in range(min(3, inputs.size(0))):
        for img_tensor, ax, title in [(inputs[i][:3], axes[i,0], "Frame T-1"), (inputs[i][3:], axes[i,1], "Frame T"), (preds[i], axes[i,2], "Prédiction T+1")]:
            ax.imshow(img_tensor.cpu().permute(1, 2, 0)); ax.set_title(title); ax.axis("off")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    plt.savefig(Path(output_dir) / f"epoch_{epoch:04d}.png")
    plt.close(fig)

# --- 5. Point d'Entrée Principal ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîner ou générer des séquences de frames avec un U-Net amélioré.")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "generate"], help="Mode: 'train' ou 'generate'.")
    train_args = parser.add_argument_group('Arguments pour l\'entraînement')
    train_args.add_argument("--data_dir", type=str, help="Dossier des frames pour l'entraînement.")
    train_args.add_argument("--epochs", type=int, default=100, help="Nombre d'époques.")
    train_args.add_argument("--batch_size", type=int, default=4, help="Taille du batch.")
    train_args.add_argument("--lr", type=float, default=1e-4, help="Taux d'apprentissage.")
    train_args.add_argument("--save_every", type=int, default=10, help="Fréquence de sauvegarde des exemples.")
    gen_args = parser.add_argument_group('Arguments pour la génération')
    gen_args.add_argument("--model_path", type=str, help="Chemin vers le modèle (.pth) pour la génération.")
    gen_args.add_argument("--start_frame_1", type=str, help="Première image de la séquence.")
    gen_args.add_argument("--start_frame_2", type=str, help="(Optionnel) Deuxième image.")
    gen_args.add_argument("--num_frames", type=int, default=10, help="Nombre de frames à générer.")
    parser.add_argument("--image_size", type=int, default=256, help="Taille des images.")
    parser.add_argument("--out_dir", type=str, default="outputs_final", help="Dossier de sortie.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_final", help="Dossier des modèles.")
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.mode == "train":
        if not args.data_dir: parser.error("--data_dir est requis pour le mode 'train'.")
        train(args, device)
    elif args.mode == "generate":
        if not args.model_path or not args.start_frame_1: parser.error("--model_path et --start_frame_1 sont requis pour le mode 'generate'.")
        generate_sequence(args, device)