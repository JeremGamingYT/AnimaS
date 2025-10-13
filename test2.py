"""
AnimeGenesis-X: Architecture Révolutionnaire pour Génération d'Anime
Développé par: AI Research Lab
Version: 1.0 EXPERIMENTAL
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

# ================== PARTIE 1: DATASET LOADER POUR ANIME ==================

class AnimeDataset(Dataset):
    """Dataset personnalisé pour charger tes images d'anime"""
    
    def __init__(self, root_dir, image_size=256, augment=True):
        """
        root_dir: Dossier contenant tes images d'anime
        image_size: Taille de redimensionnement
        augment: Appliquer des augmentations
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.augment = augment
        
        # Chercher toutes les images
        self.image_paths = []
        for ext in ['*.jpg', '*.png', '*.jpeg', '*.webp']:
            self.image_paths.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
        
        print(f"📁 Dataset chargé: {len(self.image_paths)} images trouvées")
        
        # Transformations de base
        self.base_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Augmentations pour améliorer la génération
        self.augment_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            T.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        ])
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Charger l'image
        image = Image.open(img_path).convert('RGB')
        
        # Appliquer augmentations si activé
        if self.augment and np.random.random() > 0.5:
            image = self.augment_transform(image)
        
        # Transformation de base
        image = self.base_transform(image)
        
        return image

# ================== PARTIE 2: ARCHITECTURE RÉVOLUTIONNAIRE ==================

class StyleMemoryBank(nn.Module):
    """Banque de mémoire de styles pour capturer l'essence de l'anime"""
    
    def __init__(self, num_styles=256, style_dim=512):
        super().__init__()
        self.num_styles = num_styles
        self.style_dim = style_dim
        
        # Mémoire de styles appris
        self.style_bank = nn.Parameter(torch.randn(num_styles, style_dim))
        self.style_importance = nn.Parameter(torch.ones(num_styles))
        
        # Réseau d'extraction de style
        self.style_encoder = nn.Sequential(
            nn.Linear(style_dim, style_dim * 2),
            nn.LayerNorm(style_dim * 2),
            nn.GELU(),
            nn.Linear(style_dim * 2, style_dim),
        )
        
    def forward(self, x, return_weights=False):
        """Récupère et mixe les styles pertinents"""
        B, C = x.shape
        
        # Encoder le style d'entrée
        encoded_style = self.style_encoder(x)
        
        # Calculer l'attention sur la banque de styles
        attention = torch.matmul(encoded_style, self.style_bank.T) / np.sqrt(self.style_dim)
        attention_weights = F.softmax(attention * self.style_importance, dim=-1)
        
        # Récupérer le style mixé
        mixed_style = torch.matmul(attention_weights, self.style_bank)
        
        if return_weights:
            return mixed_style, attention_weights
        return mixed_style

class WaveletTransformLayer(nn.Module):
    """Couche de transformation en ondelettes pour capturer les détails multi-échelles"""
    
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        
        # Filtres d'ondelettes apprenables
        self.wavelet_filters = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=k, padding=k//2, groups=in_channels)
            for k in [3, 5, 7, 9]
        ])
        
        self.fusion = nn.Conv2d(in_channels * 4, in_channels, 1)
        
    def forward(self, x):
        # Décomposition multi-échelle
        wavelets = []
        for filt in self.wavelet_filters:
            wavelets.append(filt(x))
        
        # Fusion adaptative
        fused = torch.cat(wavelets, dim=1)
        return self.fusion(fused) + x

class AnimeStyleAttention(nn.Module):
    """Attention spécialisée pour les caractéristiques d'anime"""
    
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        # Projections spécialisées pour l'anime
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        # Modulation de style anime
        self.style_modulation = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, style=None):
        B, N, C = x.shape
        
        # Projections Q, K, V
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention avec échelle adaptative
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Application sur les valeurs
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        
        # Modulation par le style si fourni
        if style is not None:
            style_gate = self.style_modulation(style).unsqueeze(1)
            out = out * style_gate
        
        return self.out_proj(out)

class ContourAwareConvolution(nn.Module):
    """Convolution consciente des contours pour préserver les lignes d'anime"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.main_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        
        # Détecteurs de contours spécialisés
        self.edge_x = nn.Conv2d(in_channels, out_channels//4, 1)
        self.edge_y = nn.Conv2d(in_channels, out_channels//4, 1)
        self.edge_diag1 = nn.Conv2d(in_channels, out_channels//4, 1)
        self.edge_diag2 = nn.Conv2d(in_channels, out_channels//4, 1)
        
        # Fusion
        self.edge_fusion = nn.Conv2d(out_channels, out_channels, 1)
        
    def forward(self, x):
        # Convolution principale
        main = self.main_conv(x)
        
        # Détection des contours
        dx = self.edge_x(F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0)))
        dy = self.edge_y(F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1)))
        dd1 = self.edge_diag1(F.pad(x[:, :, 1:, 1:] - x[:, :, :-1, :-1], (0, 1, 0, 1)))
        dd2 = self.edge_diag2(F.pad(x[:, :, 1:, :-1] - x[:, :, :-1, 1:], (1, 0, 0, 1)))
        
        # Combinaison
        edges = torch.cat([dx, dy, dd1, dd2], dim=1)
        edge_aware = self.edge_fusion(edges)
        
        return main + 0.3 * edge_aware

class AnimeGeneratorBlock(nn.Module):
    """Bloc générateur optimisé pour l'anime"""
    
    def __init__(self, in_channels, out_channels, style_dim=512, upsample=False):
        super().__init__()
        self.upsample = upsample
        
        # Upsampling si nécessaire
        if upsample:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        else:
            self.up = None
            
        # Convolutions conscientes des contours
        self.conv1 = ContourAwareConvolution(in_channels if not upsample else out_channels, out_channels)
        self.conv2 = ContourAwareConvolution(out_channels, out_channels)
        
        # Normalisation adaptative au style
        self.norm1 = nn.InstanceNorm2d(out_channels, affine=False)
        self.norm2 = nn.InstanceNorm2d(out_channels, affine=False)
        
        # Modulation par le style
        self.style_mod1 = nn.Linear(style_dim, out_channels * 2)
        self.style_mod2 = nn.Linear(style_dim, out_channels * 2)
        
        # Transformation en ondelettes
        self.wavelet = WaveletTransformLayer(out_channels)
        
        # Activation
        self.act = nn.LeakyReLU(0.2)
        
    def forward(self, x, style):
        # Upsampling
        if self.up is not None:
            x = self.up(x)
            
        # Première convolution avec modulation de style
        h = self.conv1(x)
        h = self.norm1(h)
        
        style1 = self.style_mod1(style).unsqueeze(2).unsqueeze(3)
        gamma1, beta1 = style1.chunk(2, dim=1)
        h = h * (1 + gamma1) + beta1
        h = self.act(h)
        
        # Deuxième convolution avec modulation de style
        h = self.conv2(h)
        h = self.norm2(h)
        
        style2 = self.style_mod2(style).unsqueeze(2).unsqueeze(3)
        gamma2, beta2 = style2.chunk(2, dim=1)
        h = h * (1 + gamma2) + beta2
        
        # Transformation en ondelettes pour les détails
        h = self.wavelet(h)
        h = self.act(h)
        
        return h

class AnimeGenesisX(nn.Module):
    """Architecture principale AnimeGenesis-X"""
    
    def __init__(self, latent_dim=512, style_dim=512, num_styles=256):
        super().__init__()
        
        # Configuration
        self.latent_dim = latent_dim
        self.style_dim = style_dim
        
        # Banque de mémoire de styles
        self.style_memory = StyleMemoryBank(num_styles, style_dim)
        
        # Mapping network pour le style
        self.style_mapping = nn.Sequential(
            nn.Linear(latent_dim, style_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(style_dim, style_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(style_dim, style_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(style_dim, style_dim)
        )
        
        # Générateur progressif
        self.initial_layer = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 1024, 4, 1, 0),
            nn.BatchNorm2d(1024),
            nn.ReLU(True)
        )
        
        # Blocs de génération avec attention
        self.blocks = nn.ModuleList([
            AnimeGeneratorBlock(1024, 512, style_dim, upsample=True),  # 4x4 -> 8x8
            AnimeGeneratorBlock(512, 256, style_dim, upsample=True),   # 8x8 -> 16x16
            AnimeGeneratorBlock(256, 128, style_dim, upsample=True),   # 16x16 -> 32x32
            AnimeGeneratorBlock(128, 64, style_dim, upsample=True),    # 32x32 -> 64x64
            AnimeGeneratorBlock(64, 32, style_dim, upsample=True),     # 64x64 -> 128x128
            AnimeGeneratorBlock(32, 16, style_dim, upsample=True),     # 128x128 -> 256x256
        ])
        
        # Attention aux résolutions critiques
        self.attention_32 = AnimeStyleAttention(128)
        self.attention_64 = AnimeStyleAttention(64)
        self.attention_128 = AnimeStyleAttention(32)
        
        # Couche de sortie avec préservation des couleurs anime
        self.to_rgb = nn.Sequential(
            ContourAwareConvolution(16, 8),
            nn.InstanceNorm2d(8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 3, 3, padding=1),
            nn.Tanh()
        )
        
        # Post-processing pour qualité supérieure
        self.refiner = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, z, return_intermediates=False):
        """
        z: vecteur latent [B, latent_dim]
        """
        B = z.shape[0]
        intermediates = []
        
        # Mapping vers l'espace de style
        w = self.style_mapping(z)
        
        # Enrichissement avec la banque de mémoire
        w_enriched = self.style_memory(w)
        
        # Génération initiale
        x = z.view(B, self.latent_dim, 1, 1)
        x = self.initial_layer(x)
        
        # Passage par les blocs avec modulation de style
        for i, block in enumerate(self.blocks):
            x = block(x, w_enriched)
            
            # Attention aux résolutions importantes
            if i == 3:  # 32x32
                B, C, H, W = x.shape
                x_flat = x.flatten(2).transpose(1, 2)
                x_att = self.attention_32(x_flat, w_enriched)
                x = x + x_att.transpose(1, 2).reshape(B, C, H, W) * 0.1
                
            elif i == 4:  # 64x64
                B, C, H, W = x.shape
                x_flat = x.flatten(2).transpose(1, 2)
                x_att = self.attention_64(x_flat, w_enriched)
                x = x + x_att.transpose(1, 2).reshape(B, C, H, W) * 0.1
                
            elif i == 5:  # 128x128
                B, C, H, W = x.shape
                x_flat = x.flatten(2).transpose(1, 2)
                x_att = self.attention_128(x_flat, w_enriched)
                x = x + x_att.transpose(1, 2).reshape(B, C, H, W) * 0.1
            
            if return_intermediates:
                intermediates.append(x)
        
        # Conversion en RGB
        rgb = self.to_rgb(x)
        
        # Raffinement final
        rgb_refined = rgb + 0.1 * self.refiner(rgb)
        
        if return_intermediates:
            return rgb_refined, intermediates
        return rgb_refined

class AnimeDiscriminator(nn.Module):
    """Discriminateur spécialisé pour l'anime"""
    
    def __init__(self):
        super().__init__()
        
        # Extracteur de caractéristiques multi-échelles
        self.features = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 32, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, 4, stride=2, padding=1),
                nn.InstanceNorm2d(64),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, 4, stride=2, padding=1),
                nn.InstanceNorm2d(128),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, 4, stride=2, padding=1),
                nn.InstanceNorm2d(256),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                nn.Conv2d(256, 512, 4, stride=2, padding=1),
                nn.InstanceNorm2d(512),
                nn.LeakyReLU(0.2),
            ),
        ])
        
        # Classificateur final
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        features = []
        h = x
        
        for layer in self.features:
            h = layer(h)
            features.append(h)
        
        # Classification
        validity = self.classifier(h)
        
        return validity, features

# ================== PARTIE 3: SYSTÈME D'ENTRAÎNEMENT AVANCÉ ==================

class AnimeTrainer:
    """Système d'entraînement complet pour AnimeGenesis-X"""
    
    def __init__(self, generator, discriminator, dataset_path, device='cuda'):
        self.G = generator.to(device)
        self.D = discriminator.to(device)
        self.device = device
        
        # Optimiseurs avec paramètres optimisés
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Dataset
        self.dataset = AnimeDataset(dataset_path, image_size=256)
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=8,  # Ajuste selon ta VRAM
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Métriques
        self.losses = {'G': [], 'D': [], 'D_real': [], 'D_fake': []}
        
        # Créer dossiers de sortie
        os.makedirs('outputs', exist_ok=True)
        os.makedirs('checkpoints', exist_ok=True)
        
    def train_step(self, real_images):
        """Une étape d'entraînement"""
        B = real_images.size(0)
        real_images = real_images.to(self.device)
        
        # Labels
        real_label = torch.ones(B, 1).to(self.device)
        fake_label = torch.zeros(B, 1).to(self.device)
        
        # ============= Entraîner le Discriminateur =============
        self.opt_D.zero_grad()
        
        # Real images
        real_validity, real_features = self.D(real_images)
        d_real_loss = F.binary_cross_entropy_with_logits(real_validity, real_label)
        
        # Fake images
        z = torch.randn(B, self.G.latent_dim).to(self.device)
        fake_images = self.G(z)
        fake_validity, fake_features = self.D(fake_images.detach())
        d_fake_loss = F.binary_cross_entropy_with_logits(fake_validity, fake_label)
        
        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        self.opt_D.step()
        
        # ============= Entraîner le Générateur =============
        self.opt_G.zero_grad()
        
        # Générer de nouvelles images
        z = torch.randn(B, self.G.latent_dim).to(self.device)
        gen_images = self.G(z)
        validity, gen_features = self.D(gen_images)
        
        # Adversarial loss
        g_adv_loss = F.binary_cross_entropy_with_logits(validity, real_label)
        
        # Feature matching loss (pour stabilité)
        fm_loss = 0
        for rf, gf in zip(real_features, gen_features):
            fm_loss += F.l1_loss(gf, rf.detach()) * 0.01
        
        # Total generator loss
        g_loss = g_adv_loss + fm_loss
        g_loss.backward()
        self.opt_G.step()
        
        # Enregistrer les losses
        self.losses['G'].append(g_loss.item())
        self.losses['D'].append(d_loss.item())
        self.losses['D_real'].append(d_real_loss.item())
        self.losses['D_fake'].append(d_fake_loss.item())
        
        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'd_real': d_real_loss.item(),
            'd_fake': d_fake_loss.item()
        }
    
    def train(self, num_epochs=100):
        """Boucle d'entraînement principale"""
        print("🚀 Début de l'entraînement AnimeGenesis-X")
        print("=" * 50)
        
        fixed_z = torch.randn(16, self.G.latent_dim).to(self.device)
        
        for epoch in range(num_epochs):
            epoch_losses = {'G': [], 'D': []}
            
            with tqdm(self.dataloader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
                for i, real_images in enumerate(pbar):
                    losses = self.train_step(real_images)
                    
                    epoch_losses['G'].append(losses['g_loss'])
                    epoch_losses['D'].append(losses['d_loss'])
                    
                    # Mise à jour de la barre de progression
                    pbar.set_postfix({
                        'G': f"{losses['g_loss']:.4f}",
                        'D': f"{losses['d_loss']:.4f}",
                        'D_r': f"{losses['d_real']:.4f}",
                        'D_f': f"{losses['d_fake']:.4f}"
                    })
                    
                    # Sauvegarder des échantillons périodiquement
                    if i % 100 == 0:
                        self.save_samples(fixed_z, epoch, i)
            
            # Statistiques d'époque
            avg_g = np.mean(epoch_losses['G'])
            avg_d = np.mean(epoch_losses['D'])
            print(f"\n📊 Epoch {epoch+1} - G Loss: {avg_g:.4f}, D Loss: {avg_d:.4f}")
            
            # Sauvegarder le modèle
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch + 1)
            
            # Graphiques de progression
            if (epoch + 1) % 5 == 0:
                self.plot_losses()
    
    @torch.no_grad()
    def save_samples(self, fixed_z, epoch, step):
        """Sauvegarder des échantillons générés"""
        self.G.eval()
        
        fake_images = self.G(fixed_z)
        
        # Dénormaliser
        fake_images = fake_images * 0.5 + 0.5
        
        # Créer une grille
        grid = make_grid(fake_images, nrow=4, padding=2, normalize=False)
        
        # Sauvegarder
        save_path = f'outputs/epoch_{epoch:04d}_step_{step:05d}.png'
        save_image(grid, save_path)
        
        self.G.train()
    
    def save_checkpoint(self, epoch):
        """Sauvegarder un checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator': self.G.state_dict(),
            'discriminator': self.D.state_dict(),
            'opt_g': self.opt_G.state_dict(),
            'opt_d': self.opt_D.state_dict(),
            'losses': self.losses
        }
        
        path = f'checkpoints/checkpoint_epoch_{epoch:04d}.pth'
        torch.save(checkpoint, path)
        print(f"💾 Checkpoint sauvegardé: {path}")
    
    def plot_losses(self):
        """Visualiser les courbes de loss"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.losses['G'], label='Generator', alpha=0.7)
        plt.plot(self.losses['D'], label='Discriminator', alpha=0.7)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Losses')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.losses['D_real'], label='D(real)', alpha=0.7)
        plt.plot(self.losses['D_fake'], label='D(fake)', alpha=0.7)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Discriminator Details')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/losses.png', dpi=100)
        plt.close()

# ================== PARTIE 4: INTERFACE DE TEST ==================

class AnimeGenerator:
    """Interface simple pour générer des images"""
    
    def __init__(self, checkpoint_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Utilisation de: {self.device}")
        
        # Créer le modèle
        self.model = AnimeGenesisX().to(self.device)
        
        # Charger checkpoint si disponible
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['generator'])
            print(f"✅ Modèle chargé depuis: {checkpoint_path}")
        else:
            print("⚠️ Aucun checkpoint trouvé, utilisation du modèle aléatoire")
        
        self.model.eval()
    
    @torch.no_grad()
    def generate(self, num_images=1, seed=None):
        """Générer des images d'anime"""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Générer vecteurs latents
        z = torch.randn(num_images, self.model.latent_dim).to(self.device)
        
        # Générer images
        images = self.model(z)
        
        # Dénormaliser
        images = images * 0.5 + 0.5
        images = torch.clamp(images, 0, 1)
        
        return images
    
    def generate_and_save(self, num_images=4, output_dir='generated'):
        """Générer et sauvegarder des images"""
        os.makedirs(output_dir, exist_ok=True)
        
        images = self.generate(num_images)
        
        for i, img in enumerate(images):
            save_path = os.path.join(output_dir, f'anime_{i:04d}.png')
            save_image(img, save_path)
            print(f"✅ Image sauvegardée: {save_path}")
        
        # Créer aussi une grille
        grid = make_grid(images, nrow=2, padding=2)
        grid_path = os.path.join(output_dir, 'grid.png')
        save_image(grid, grid_path)
        print(f"🎨 Grille sauvegardée: {grid_path}")

# ================== SCRIPT PRINCIPAL ==================

def main():
    """Script principal pour tester l'architecture"""
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║         AnimeGenesis-X : Générateur d'Anime IA          ║
    ║                    Version 1.0 BETA                      ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    DATASET_PATH = "/kaggle/input/anima-s-dataset/test/"  # 👈 CHANGE CE CHEMIN VERS TON DATASET
    TRAIN_MODE = True  # Mettre False pour juste générer
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️ Device: {device}")
    
    if torch.cuda.is_available():
        print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    if TRAIN_MODE:
        # ========== MODE ENTRAÎNEMENT ==========
        print("\n🎯 Mode: ENTRAÎNEMENT")
        
        # Vérifier le dataset
        if not os.path.exists(DATASET_PATH):
            print(f"\n⚠️ ATTENTION: Le dossier '{DATASET_PATH}' n'existe pas!")
            print("📁 Création du dossier...")
            os.makedirs(DATASET_PATH)
            print(f"👉 Place tes images d'anime dans: {os.path.abspath(DATASET_PATH)}")
            print("   Formats supportés: .jpg, .png, .jpeg, .webp")
            return
        
        # Créer les modèles
        print("\n🔨 Création des modèles...")
        generator = AnimeGenesisX(latent_dim=512, style_dim=512, num_styles=256)
        discriminator = AnimeDiscriminator()
        
        # Compter les paramètres
        g_params = sum(p.numel() for p in generator.parameters()) / 1e6
        d_params = sum(p.numel() for p in discriminator.parameters()) / 1e6
        print(f"📊 Générateur: {g_params:.2f}M paramètres")
        print(f"📊 Discriminateur: {d_params:.2f}M paramètres")
        
        # Créer le trainer
        trainer = AnimeTrainer(generator, discriminator, DATASET_PATH, device)
        
        # Lancer l'entraînement
        print("\n🚀 Lancement de l'entraînement...")
        print("💡 Conseil: Les images seront sauvées dans 'outputs/'")
        print("⌛ Cela peut prendre plusieurs heures selon ton GPU\n")
        
        trainer.train(num_epochs=100)  # Ajuste le nombre d'époques
        
    else:
        # ========== MODE GÉNÉRATION ==========
        print("\n🎯 Mode: GÉNÉRATION")
        
        # Chercher le dernier checkpoint
        checkpoints = glob.glob('checkpoints/*.pth')
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            print(f"📁 Checkpoint trouvé: {latest_checkpoint}")
        else:
            latest_checkpoint = None
            print("⚠️ Aucun checkpoint trouvé, génération avec poids aléatoires")
        
        # Créer le générateur
        gen = AnimeGenerator(latest_checkpoint)
        
        # Générer des images
        print("\n🎨 Génération d'images d'anime...")
        gen.generate_and_save(num_images=16)
        
        print("\n✅ Génération terminée! Vérifie le dossier 'generated/'")
    
    print("\n🌟 Terminé!")

if __name__ == "__main__":
    main()