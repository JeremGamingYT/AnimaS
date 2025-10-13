"""
AnimeGenesis-X: Architecture Optimisée pour 12GB VRAM
Version: 1.2 MEMORY_OPTIMIZED
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image, make_grid
import numpy as np
from PIL import Image
import os
import glob
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

# Configuration optimisée mémoire
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# ================== DATASET OPTIMISÉ ==================

class AnimeDataset(Dataset):
    """Dataset optimisé pour économiser la mémoire"""
    
    def __init__(self, root_dir, image_size=128, augment=True):  # Réduit à 128
        self.root_dir = root_dir
        self.image_size = image_size
        self.augment = augment
        
        self.image_paths = []
        for ext in ['*.jpg', '*.png', '*.jpeg', '*.webp']:
            self.image_paths.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
        
        print(f"📁 Dataset: {len(self.image_paths)} images trouvées")
        
        if len(self.image_paths) == 0:
            print("⚠️ Création d'images de test...")
            self.create_dummy_images = True
        else:
            self.create_dummy_images = False
        
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return max(len(self.image_paths), 100)
    
    def __getitem__(self, idx):
        if self.create_dummy_images or len(self.image_paths) == 0:
            img_array = np.random.rand(self.image_size, self.image_size, 3) * 255
            image = Image.fromarray(img_array.astype('uint8'), 'RGB')
        else:
            img_path = self.image_paths[idx % len(self.image_paths)]
            image = Image.open(img_path).convert('RGB')
        
        return self.transform(image)

# ================== MODULES OPTIMISÉS ==================

class EfficientStyleBank(nn.Module):
    """Banque de styles optimisée en mémoire"""
    
    def __init__(self, num_styles=64, style_dim=256):  # Réduit de 256->64 styles, 512->256 dim
        super().__init__()
        self.num_styles = num_styles
        self.style_dim = style_dim
        
        # Banque réduite
        self.style_bank = nn.Parameter(torch.randn(num_styles, style_dim) * 0.02)
        
        # Réseau plus léger
        self.style_encoder = nn.Sequential(
            nn.Linear(style_dim, style_dim),
            nn.GELU(),
            nn.Linear(style_dim, style_dim),
        )
    
    def forward(self, x):
        encoded = self.style_encoder(x)
        attention = F.softmax(torch.matmul(encoded, self.style_bank.T) / np.sqrt(self.style_dim), dim=-1)
        return torch.matmul(attention, self.style_bank)

class LightweightWavelet(nn.Module):
    """Version allégée de la transformation en ondelettes"""
    
    def __init__(self, channels):
        super().__init__()
        # Utiliser seulement 2 échelles au lieu de 4
        self.conv1 = nn.Conv2d(channels, channels//2, 3, padding=1, groups=min(channels//2, 8))
        self.conv2 = nn.Conv2d(channels, channels//2, 5, padding=2, groups=min(channels//2, 8))
        self.fusion = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        # Application séquentielle pour économiser la mémoire
        w1 = self.conv1(x)
        w2 = self.conv2(x)
        combined = torch.cat([w1, w2], dim=1)
        return x + 0.1 * self.fusion(combined)

class MemoryEfficientAttention(nn.Module):
    """Attention optimisée pour économiser la mémoire"""
    
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = min(num_heads, dim // 16)
        self.dim = dim
        self.head_dim = dim // self.num_heads
        
        # Une seule projection pour économiser
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, style=None):
        B, N, C = x.shape
        
        # QKV en une seule passe
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention avec checkpointing si possible
        with torch.cuda.amp.autocast(enabled=False):  # Désactiver temporairement pour cette op
            attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        return self.proj(out)

class OptimizedGeneratorBlock(nn.Module):
    """Bloc générateur optimisé en mémoire"""
    
    def __init__(self, in_channels, out_channels, style_dim=256):
        super().__init__()
        
        # Upsampling plus efficace
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        # Convolutions légères
        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Normalisation
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        
        # Style modulation simplifiée
        self.style_mod = nn.Linear(style_dim, out_channels * 2)
        
        self.act = nn.LeakyReLU(0.2, inplace=True)  # inplace pour économiser
    
    def forward(self, x, style):
        # Upsampling
        x = self.up(x)
        x = self.conv(x)
        
        # Première conv
        h = self.norm1(self.conv1(x))
        style_params = self.style_mod(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style_params.chunk(2, dim=1)
        h = h * (1 + gamma * 0.1) + beta * 0.1  # Réduire l'impact
        h = self.act(h)
        
        # Deuxième conv avec skip connection
        h = self.norm2(self.conv2(h))
        h = self.act(h + x * 0.1)  # Skip connection légère
        
        return h

# ================== GÉNÉRATEUR OPTIMISÉ ==================

class AnimeGenesisXLite(nn.Module):
    """Version allégée pour 12GB VRAM"""
    
    def __init__(self, latent_dim=256, style_dim=256, num_styles=64):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.style_dim = style_dim
        
        # Banque de mémoire légère
        self.style_memory = EfficientStyleBank(num_styles, style_dim)
        
        # Mapping simplifié
        self.style_mapping = nn.Sequential(
            nn.Linear(latent_dim, style_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(style_dim, style_dim)
        )
        
        # Démarrage avec moins de canaux
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0),  # 1024 -> 512
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        
        # Blocs optimisés
        self.blocks = nn.ModuleList([
            OptimizedGeneratorBlock(512, 256, style_dim),  # 4x4 -> 8x8
            OptimizedGeneratorBlock(256, 128, style_dim),  # 8x8 -> 16x16
            OptimizedGeneratorBlock(128, 64, style_dim),   # 16x16 -> 32x32
            OptimizedGeneratorBlock(64, 32, style_dim),    # 32x32 -> 64x64
            OptimizedGeneratorBlock(32, 16, style_dim),    # 64x64 -> 128x128
        ])
        
        # Une seule attention légère
        self.attention = MemoryEfficientAttention(32, num_heads=2)
        
        # Sortie simplifiée
        self.to_rgb = nn.Sequential(
            nn.Conv2d(16, 8, 3, padding=1),
            nn.GroupNorm(4, 8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 3, 3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z):
        B = z.shape[0]
        
        # Style mapping
        w = self.style_mapping(z)
        w = self.style_memory(w)
        
        # Génération
        x = z.view(B, self.latent_dim, 1, 1)
        x = self.initial(x)
        
        for i, block in enumerate(self.blocks):
            x = block(x, w)
            
            # Attention seulement à une résolution
            if i == 3:  # 64x64
                B, C, H, W = x.shape
                x_flat = x.flatten(2).transpose(1, 2)
                x_att = self.attention(x_flat)
                x = x + x_att.transpose(1, 2).reshape(B, C, H, W) * 0.05
        
        # RGB
        return self.to_rgb(x)

class LightDiscriminator(nn.Module):
    """Discriminateur allégé"""
    
    def __init__(self):
        super().__init__()
        
        self.features = nn.Sequential(
            # 128x128 -> 64x64
            nn.Conv2d(3, 16, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64x64 -> 32x32
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.GroupNorm(4, 32),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 -> 16x16
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 -> 4x4
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        features = self.features(x)
        return self.classifier(features), [features]

# ================== TRAINER OPTIMISÉ ==================

class OptimizedTrainer:
    """Trainer optimisé pour économiser la mémoire"""
    
    def __init__(self, generator, discriminator, dataset_path, device='cuda'):
        self.G = generator.to(device)
        self.D = discriminator.to(device)
        self.device = device
        
        # Learning rate plus bas pour stabilité
        self.opt_G = torch.optim.AdamW(self.G.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.opt_D = torch.optim.AdamW(self.D.parameters(), lr=0.0001, betas=(0.5, 0.999))
        
        # Dataset avec images plus petites
        self.dataset = AnimeDataset(dataset_path, image_size=128)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=2,  # Batch size minimal
            shuffle=True,
            num_workers=0,  # 0 pour économiser la RAM
            pin_memory=False
        )
        
        # Créer dossiers
        os.makedirs('outputs', exist_ok=True)
        os.makedirs('checkpoints', exist_ok=True)
        
        self.losses = {'G': [], 'D': []}
        self.iteration = 0
        
        # Mixed precision pour économiser la mémoire
        self.scaler_G = torch.cuda.amp.GradScaler()
        self.scaler_D = torch.cuda.amp.GradScaler()
    
    def train_step(self, real_images):
        """Étape d'entraînement optimisée"""
        B = real_images.size(0)
        real_images = real_images.to(self.device)
        
        # Labels
        real_label = torch.ones(B, 1).to(self.device) * 0.9
        fake_label = torch.zeros(B, 1).to(self.device) + 0.1
        
        # ========= Discriminateur =========
        self.opt_D.zero_grad(set_to_none=True)  # Plus efficace
        
        with torch.cuda.amp.autocast():
            # Real
            real_validity, _ = self.D(real_images)
            d_real_loss = F.binary_cross_entropy_with_logits(real_validity, real_label)
            
            # Fake
            z = torch.randn(B, self.G.latent_dim).to(self.device)
            fake_images = self.G(z)
            fake_validity, _ = self.D(fake_images.detach())
            d_fake_loss = F.binary_cross_entropy_with_logits(fake_validity, fake_label)
            
            d_loss = (d_real_loss + d_fake_loss) / 2
        
        self.scaler_D.scale(d_loss).backward()
        self.scaler_D.unscale_(self.opt_D)
        torch.nn.utils.clip_grad_norm_(self.D.parameters(), 1.0)
        self.scaler_D.step(self.opt_D)
        self.scaler_D.update()
        
        # ========= Générateur =========
        self.opt_G.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast():
            z = torch.randn(B, self.G.latent_dim).to(self.device)
            gen_images = self.G(z)
            validity, _ = self.D(gen_images)
            g_loss = F.binary_cross_entropy_with_logits(validity, real_label)
        
        self.scaler_G.scale(g_loss).backward()
        self.scaler_G.unscale_(self.opt_G)
        torch.nn.utils.clip_grad_norm_(self.G.parameters(), 1.0)
        self.scaler_G.step(self.opt_G)
        self.scaler_G.update()
        
        # Libérer la mémoire
        del fake_images, gen_images
        torch.cuda.empty_cache()
        
        self.losses['G'].append(g_loss.item())
        self.losses['D'].append(d_loss.item())
        self.iteration += 1
        
        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item()
        }
    
    def train(self, num_epochs=50):
        """Entraînement optimisé"""
        print("🚀 Début de l'entraînement (Mode Économie Mémoire)")
        
        fixed_z = torch.randn(4, self.G.latent_dim).to(self.device)
        
        for epoch in range(num_epochs):
            pbar = tqdm(self.dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            for real_images in pbar:
                if real_images.size(0) < 2:
                    continue
                
                losses = self.train_step(real_images)
                
                pbar.set_postfix({
                    'G': f"{losses['g_loss']:.4f}",
                    'D': f"{losses['d_loss']:.4f}"
                })
                
                # Sauvegarder moins souvent
                if self.iteration % 100 == 0:
                    self.save_samples(fixed_z, epoch)
            
            # Checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch + 1)
                
            # Nettoyer la mémoire
            torch.cuda.empty_cache()
    
    @torch.no_grad()
    def save_samples(self, fixed_z, epoch):
        """Sauvegarder des échantillons"""
        self.G.eval()
        
        fake = self.G(fixed_z)
        fake = fake * 0.5 + 0.5
        
        save_image(fake, f'outputs/epoch_{epoch:04d}_iter_{self.iteration:06d}.png', nrow=2)
        
        self.G.train()
    
    def save_checkpoint(self, epoch):
        """Sauvegarder checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator': self.G.state_dict(),
            'discriminator': self.D.state_dict(),
            'opt_g': self.opt_G.state_dict(),
            'opt_d': self.opt_D.state_dict(),
        }
        
        torch.save(checkpoint, f'checkpoints/checkpoint_epoch_{epoch:04d}.pth')
        print(f"💾 Checkpoint sauvegardé")

# ================== MAIN ==================

def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     AnimeGenesis-X LITE : Optimisé pour 12GB VRAM      ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        # Optimisations mémoire
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        
        print(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Configuration
    DATASET_PATH = "/kaggle/input/anima-s-dataset/test/"  # Change ce chemin
    
    # Créer dossier si nécessaire
    os.makedirs(DATASET_PATH, exist_ok=True)
    
    # Créer modèles allégés
    print("\n🔨 Création des modèles optimisés...")
    generator = AnimeGenesisXLite(latent_dim=256, style_dim=256, num_styles=64)
    discriminator = LightDiscriminator()
    
    # Stats
    g_params = sum(p.numel() for p in generator.parameters()) / 1e6
    d_params = sum(p.numel() for p in discriminator.parameters()) / 1e6
    print(f"📊 Générateur: {g_params:.2f}M paramètres (optimisé)")
    print(f"📊 Discriminateur: {d_params:.2f}M paramètres (optimisé)")
    
    # Entraînement
    trainer = OptimizedTrainer(generator, discriminator, DATASET_PATH, device)
    
    print("\n🚀 Lancement...")
    trainer.train(num_epochs=50)
    
    print("\n✅ Terminé!")

if __name__ == "__main__":
    main()