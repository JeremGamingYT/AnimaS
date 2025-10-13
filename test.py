import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import random
from datetime import datetime
import lpips
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# ==================== Configuration ====================
class Config:
    # Chemins
    FRAMES_DIR = "/kaggle/input/anima-s-dataset/test"
    CHECKPOINT_DIR = "checkpoints"
    OUTPUT_DIR = "predictions"
    
    # Hyperparamètres
    SEQUENCE_LENGTH = 5
    IMG_SIZE = (256, 256)
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4  # Pour simuler un batch plus grand
    LEARNING_RATE_G = 0.0002  # Generator
    LEARNING_RATE_D = 0.0001  # Discriminator
    NUM_EPOCHS = 200
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model
    HIDDEN_DIM = 512
    NUM_LAYERS = 3
    VIT_DIM = 768
    VIT_DEPTH = 12
    VIT_HEADS = 12
    VIT_MLP_DIM = 3072
    
    # Loss weights
    L1_WEIGHT = 100.0
    PERCEPTUAL_WEIGHT = 10.0
    GAN_WEIGHT = 1.0
    TEMPORAL_WEIGHT = 5.0
    
    # Training
    USE_MIXED_PRECISION = True
    CLIP_GRAD_NORM = 1.0

# ==================== Vision Transformer Block ====================
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.attn = MultiHeadAttention(dim, heads, dim_head, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = self.attn(self.norm1(x)) + x
        x = self.mlp(self.norm2(x)) + x
        return x

# ==================== Advanced Dataset with Augmentation ====================
class EnhancedAnimeDataset(Dataset):
    def __init__(self, frames_dir, sequence_length=5, transform=None, augment=True):
        self.frames_dir = Path(frames_dir)
        self.sequence_length = sequence_length
        self.augment = augment
        
        # Transformations de base
        self.base_transform = transforms.Compose([
            transforms.Resize(Config.IMG_SIZE),
            transforms.ToTensor()
        ])
        
        # Augmentation
        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        ]) if augment else None
        
        # Temporal augmentation
        self.temporal_augment = augment
        
        self.frame_paths = sorted(self.frames_dir.glob("frame_*.png"))
        self.num_sequences = len(self.frame_paths) - sequence_length
        
        if self.num_sequences <= 0:
            raise ValueError(f"Pas assez de frames. Minimum requis: {sequence_length + 1}")
    
    def apply_temporal_augmentation(self, frames):
        """Applique des augmentations temporelles cohérentes"""
        if not self.temporal_augment or random.random() > 0.5:
            return frames
        
        # Speed variation (frame skipping/interpolation)
        speed_factor = random.uniform(0.8, 1.2)
        
        # Pour simplifier, on retourne les frames normales
        # Dans une implémentation complète, on pourrait interpoler
        return frames
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        # Charger les frames
        input_frames = []
        
        # Paramètres d'augmentation cohérents pour la séquence
        if self.augment:
            seed = random.randint(0, 2**32)
        
        for i in range(self.sequence_length):
            frame_path = self.frame_paths[idx + i]
            frame = Image.open(frame_path).convert('RGB')
            
            # Augmentation cohérente
            if self.augmentation and self.augment:
                random.seed(seed)
                torch.manual_seed(seed)
                frame = self.augmentation(frame)
            
            frame = self.base_transform(frame)
            input_frames.append(frame)
        
        # Frame cible
        target_frame_path = self.frame_paths[idx + self.sequence_length]
        target_frame = Image.open(target_frame_path).convert('RGB')
        
        if self.augmentation and self.augment:
            random.seed(seed)
            torch.manual_seed(seed)
            target_frame = self.augmentation(target_frame)
        
        target_frame = self.base_transform(target_frame)
        
        input_tensor = torch.stack(input_frames)
        
        return input_tensor, target_frame

# ==================== Generator avec Vision Transformer ====================
class HybridGenerator(nn.Module):
    """Générateur hybride CNN + Vision Transformer"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Encoder CNN pour extraction de features
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        
        # Temporal ConvLSTM
        self.convlstm = ConvLSTM(512, 512, kernel_size=3, num_layers=2)
        
        # Vision Transformer pour capturer les dépendances globales
        self.patch_size = 16
        self.num_patches = (config.IMG_SIZE[0] // 16) * (config.IMG_SIZE[1] // 16)
        
        self.patch_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                     p1=self.patch_size, p2=self.patch_size),
            nn.Linear(self.patch_size * self.patch_size * 512, config.VIT_DIM)
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, config.VIT_DIM))
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                config.VIT_DIM, 
                config.VIT_HEADS, 
                config.VIT_DIM // config.VIT_HEADS,
                config.VIT_MLP_DIM
            ) for _ in range(config.VIT_DEPTH)
        ])
        
        self.norm = nn.LayerNorm(config.VIT_DIM)
        
        # Unpatch
        self.unpatch = nn.Sequential(
            nn.Linear(config.VIT_DIM, self.patch_size * self.patch_size * 512),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                     h=config.IMG_SIZE[0]//self.patch_size,
                     w=config.IMG_SIZE[1]//self.patch_size,
                     p1=self.patch_size, p2=self.patch_size)
        )
        
        # Decoder avec skip connections
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        
        # Encoder chaque frame
        encoded_frames = []
        for t in range(seq_len):
            encoded = self.encoder(x[:, t])
            encoded_frames.append(encoded)
        
        # Stack pour ConvLSTM
        encoded_sequence = torch.stack(encoded_frames, dim=1)
        
        # Process avec ConvLSTM
        lstm_out, _ = self.convlstm(encoded_sequence)
        last_hidden = lstm_out[:, -1]  # Prendre la dernière sortie
        
        # Vision Transformer pour améliorer les features
        # Reshape pour patches
        b, c, h_feat, w_feat = last_hidden.shape
        
        # Si les dimensions ne correspondent pas, ajuster
        if h_feat * w_feat != self.num_patches:
            # Redimensionner pour correspondre aux patches attendus
            last_hidden = F.interpolate(last_hidden, size=(16, 16), mode='bilinear')
        
        # Patch embedding
        patches = self.patch_embed(last_hidden)
        patches = patches + self.pos_embedding
        
        # Transformer blocks
        for transformer in self.transformer_blocks:
            patches = transformer(patches)
        
        patches = self.norm(patches)
        
        # Unpatch
        features = self.unpatch(patches)
        
        # Decoder
        output = self.decoder(features)
        
        return output

# ==================== ConvLSTM Implementation ====================
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )
    
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        layers = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim
            layers.append(ConvLSTMCell(cur_input_dim, hidden_dim, kernel_size))
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size()
        
        # Initialize hidden states
        h = [None] * self.num_layers
        c = [None] * self.num_layers
        
        outputs = []
        
        for t in range(seq_len):
            input_t = x[:, t]
            
            for layer_idx, layer in enumerate(self.layers):
                if h[layer_idx] is None:
                    h[layer_idx] = torch.zeros(
                        batch_size, self.hidden_dim, height, width, 
                        device=x.device
                    )
                    c[layer_idx] = torch.zeros_like(h[layer_idx])
                
                if layer_idx == 0:
                    h[layer_idx], c[layer_idx] = layer(
                        input_t, (h[layer_idx], c[layer_idx])
                    )
                else:
                    h[layer_idx], c[layer_idx] = layer(
                        h[layer_idx-1], (h[layer_idx], c[layer_idx])
                    )
            
            outputs.append(h[-1])
        
        return torch.stack(outputs, dim=1), (h, c)

# ==================== Discriminator ====================
class PatchGANDiscriminator(nn.Module):
    """PatchGAN Discriminator pour juger le réalisme local"""
    
    def __init__(self, input_channels=3):
        super().__init__()
        
        def discriminator_block(in_filters, out_filters, stride=2, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(input_channels * 2, 64, normalize=False),  # x2 pour input+target
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, padding=1)  # PatchGAN output
        )
    
    def forward(self, input_sequence, target):
        # Prendre la dernière frame de la séquence comme référence
        last_frame = input_sequence[:, -1] if len(input_sequence.shape) == 5 else input_sequence
        
        # Concatener avec la cible
        combined = torch.cat([last_frame, target], dim=1)
        output = self.model(combined)
        return output

# ==================== Loss Functions ====================
class PerceptualLoss(nn.Module):
    """Loss perceptuelle utilisant VGG"""
    
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features.to(device).eval()
        
        self.blocks = nn.ModuleList([
            vgg[:4],   # relu1_2
            vgg[4:9],  # relu2_2
            vgg[9:18], # relu3_4
            vgg[18:27],# relu4_4
            vgg[27:36] # relu5_4
        ])
        
        for p in self.parameters():
            p.requires_grad = False
        
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
    
    def forward(self, x, y):
        loss = 0.0
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            loss += self.weights[i] * F.l1_loss(x, y)
        
        return loss

class TemporalConsistencyLoss(nn.Module):
    """Loss pour la cohérence temporelle"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, frames_sequence):
        """
        Calcule la cohérence temporelle entre frames consécutives
        """
        if len(frames_sequence) < 2:
            return torch.tensor(0.0, device=frames_sequence.device)
        
        loss = 0.0
        for i in range(len(frames_sequence) - 1):
            # Optical flow approximation
            diff = frames_sequence[i+1] - frames_sequence[i]
            loss += torch.mean(torch.abs(diff))
        
        return loss / (len(frames_sequence) - 1)

# ==================== Enhanced Trainer avec GAN ====================
class GANTrainer:
    def __init__(self, generator, discriminator, config):
        self.generator = generator.to(config.DEVICE)
        self.discriminator = discriminator.to(config.DEVICE)
        self.config = config
        
        # Optimizers
        self.optimizer_G = optim.Adam(
            generator.parameters(), 
            lr=config.LEARNING_RATE_G, 
            betas=(0.5, 0.999)
        )
        self.optimizer_D = optim.Adam(
            discriminator.parameters(), 
            lr=config.LEARNING_RATE_D, 
            betas=(0.5, 0.999)
        )
        
        # Losses
        self.criterion_GAN = nn.BCEWithLogitsLoss()
        self.criterion_L1 = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss(config.DEVICE)
        self.temporal_loss = TemporalConsistencyLoss()
        
        # LPIPS
        self.lpips_loss = lpips.LPIPS(net='alex').to(config.DEVICE)
        
        # Mixed precision
        self.scaler = GradScaler() if config.USE_MIXED_PRECISION else None
        
        # Schedulers
        self.scheduler_G = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_G, T_max=config.NUM_EPOCHS
        )
        self.scheduler_D = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_D, T_max=config.NUM_EPOCHS
        )
        
        # Metrics
        self.g_losses = []
        self.d_losses = []
        
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    def train_discriminator(self, real_batch, fake_batch):
        """Entraîner le discriminateur"""
        self.optimizer_D.zero_grad()
        
        batch_size = real_batch.size(0)
        
        # Labels
        real_labels = torch.ones(batch_size, 1, 30, 30).to(self.config.DEVICE) * 0.9  # Label smoothing
        fake_labels = torch.zeros(batch_size, 1, 30, 30).to(self.config.DEVICE) + 0.1
        
        with autocast(enabled=self.config.USE_MIXED_PRECISION):
            # Real loss
            real_pred = self.discriminator(fake_batch, real_batch)
            real_loss = self.criterion_GAN(real_pred, real_labels)
            
            # Fake loss
            fake_pred = self.discriminator(fake_batch, fake_batch.detach())
            fake_loss = self.criterion_GAN(fake_pred, fake_labels)
            
            d_loss = (real_loss + fake_loss) / 2
        
        if self.scaler:
            self.scaler.scale(d_loss).backward()
            self.scaler.step(self.optimizer_D)
            self.scaler.update()
        else:
            d_loss.backward()
            self.optimizer_D.step()
        
        return d_loss.item()
    
    def train_generator(self, input_frames, target_frame, accumulation_step):
        """Entraîner le générateur avec gradient accumulation"""
        
        batch_size = target_frame.size(0)
        
        with autocast(enabled=self.config.USE_MIXED_PRECISION):
            # Forward pass
            generated_frame = self.generator(input_frames)
            
            # Adversarial loss
            fake_pred = self.discriminator(input_frames, generated_frame)
            real_labels = torch.ones_like(fake_pred)
            gan_loss = self.criterion_GAN(fake_pred, real_labels)
            
            # L1 loss
            l1_loss = self.criterion_L1(generated_frame, target_frame)
            
            # Perceptual loss
            perc_loss = self.perceptual_loss(generated_frame, target_frame)
            
            # LPIPS loss
            lpips_val = self.lpips_loss(generated_frame, target_frame).mean()
            
            # Temporal consistency (si on a plusieurs frames générées)
            temp_loss = torch.tensor(0.0).to(self.config.DEVICE)
            
            # Total loss
            g_loss = (
                self.config.GAN_WEIGHT * gan_loss +
                self.config.L1_WEIGHT * l1_loss +
                self.config.PERCEPTUAL_WEIGHT * perc_loss +
                10.0 * lpips_val +
                self.config.TEMPORAL_WEIGHT * temp_loss
            ) / self.config.GRADIENT_ACCUMULATION_STEPS
        
        if self.scaler:
            self.scaler.scale(g_loss).backward()
            
            if (accumulation_step + 1) % self.config.GRADIENT_ACCUMULATION_STEPS == 0:
                self.scaler.unscale_(self.optimizer_G)
                torch.nn.utils.clip_grad_norm_(
                    self.generator.parameters(), 
                    self.config.CLIP_GRAD_NORM
                )
                self.scaler.step(self.optimizer_G)
                self.scaler.update()
                self.optimizer_G.zero_grad()
        else:
            g_loss.backward()
            
            if (accumulation_step + 1) % self.config.GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.generator.parameters(), 
                    self.config.CLIP_GRAD_NORM
                )
                self.optimizer_G.step()
                self.optimizer_G.zero_grad()
        
        return {
            'total': g_loss.item() * self.config.GRADIENT_ACCUMULATION_STEPS,
            'gan': gan_loss.item(),
            'l1': l1_loss.item(),
            'perceptual': perc_loss.item(),
            'lpips': lpips_val.item()
        }
    
    def train_epoch(self, dataloader, epoch):
        """Entraîner une epoch complète"""
        self.generator.train()
        self.discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}") as pbar:
            for batch_idx, (input_frames, target_frame) in enumerate(pbar):
                input_frames = input_frames.to(self.config.DEVICE)
                target_frame = target_frame.to(self.config.DEVICE)
                
                # Train Discriminator
                if batch_idx % 2 == 0:  # Train D every 2 iterations
                    with torch.no_grad():
                        fake_frame = self.generator(input_frames)
                    
                    d_loss = self.train_discriminator(target_frame, fake_frame)
                    epoch_d_loss += d_loss
                
                # Train Generator
                g_losses = self.train_generator(input_frames, target_frame, batch_idx)
                epoch_g_loss += g_losses['total']
                
                # Update progress bar
                pbar.set_postfix({
                    'G_Loss': f"{g_losses['total']:.4f}",
                    'D_Loss': f"{d_loss:.4f}" if batch_idx % 2 == 0 else "N/A",
                    'L1': f"{g_losses['l1']:.4f}",
                    'LPIPS': f"{g_losses['lpips']:.4f}"
                })
        
        # Update schedulers
        self.scheduler_G.step()
        self.scheduler_D.step()
        
        return epoch_g_loss / len(dataloader), epoch_d_loss / (len(dataloader) // 2)
    
    def validate(self, dataloader):
        """Validation"""
        self.generator.eval()
        val_loss = 0
        
        with torch.no_grad():
            for input_frames, target_frame in dataloader:
                input_frames = input_frames.to(self.config.DEVICE)
                target_frame = target_frame.to(self.config.DEVICE)
                
                generated_frame = self.generator(input_frames)
                loss = self.criterion_L1(generated_frame, target_frame)
                val_loss += loss.item()
        
        return val_loss / len(dataloader)
    
    def save_checkpoint(self, epoch, g_loss, d_loss):
        """Sauvegarder checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'g_loss': g_loss,
            'd_loss': d_loss,
            'g_losses': self.g_losses,
            'd_losses': self.d_losses
        }
        
        checkpoint_path = os.path.join(
            self.config.CHECKPOINT_DIR,
            f'checkpoint_epoch_{epoch}_gloss_{g_loss:.4f}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        print(f"✅ Checkpoint sauvegardé: {checkpoint_path}")
    
    def visualize_results(self, dataloader, epoch):
        """Visualiser les résultats"""
        self.generator.eval()
        
        with torch.no_grad():
            input_frames, target_frame = next(iter(dataloader))
            input_frames = input_frames.to(self.config.DEVICE)
            target_frame = target_frame.to(self.config.DEVICE)
            
            generated_frame = self.generator(input_frames)
            
            # Denormalize
            def denorm(x):
                return (x + 1) / 2
            
            # Convert to numpy
            input_last = denorm(input_frames[0, -1]).cpu().numpy().transpose(1, 2, 0)
            target = denorm(target_frame[0]).cpu().numpy().transpose(1, 2, 0)
            generated = denorm(generated_frame[0]).cpu().numpy().transpose(1, 2, 0)
            
            # Create figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(np.clip(input_last, 0, 1))
            axes[0].set_title("Dernière frame d'entrée")
            axes[0].axis('off')
            
            axes[1].imshow(np.clip(target, 0, 1))
            axes[1].set_title("Frame cible")
            axes[1].axis('off')
            
            axes[2].imshow(np.clip(generated, 0, 1))
            axes[2].set_title("Frame générée")
            axes[2].axis('off')
            
            plt.suptitle(f"Epoch {epoch + 1}")
            plt.tight_layout()
            
            save_path = os.path.join(
                self.config.OUTPUT_DIR,
                f'prediction_epoch_{epoch+1}.png'
            )
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"📸 Visualisation sauvegardée: {save_path}")
    
    def train(self, train_loader, val_loader=None):
        """Entraînement complet"""
        print(f"🚀 Début de l'entraînement GAN sur {self.config.DEVICE}")
        print(f"📊 Mixed Precision: {self.config.USE_MIXED_PRECISION}")
        print(f"📊 Gradient Accumulation Steps: {self.config.GRADIENT_ACCUMULATION_STEPS}")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\n{'='*50}")
            print(f"📅 Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
            print(f"{'='*50}")
            
            # Training
            g_loss, d_loss = self.train_epoch(train_loader, epoch)
            self.g_losses.append(g_loss)
            self.d_losses.append(d_loss)
            
            print(f"📉 Generator Loss: {g_loss:.4f}")
            print(f"📉 Discriminator Loss: {d_loss:.4f}")
            
            # Validation
            if val_loader:
                val_loss = self.validate(val_loader)
                print(f"📈 Validation Loss: {val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(epoch, g_loss, d_loss)
            
            # Visualize
            if (epoch + 1) % 5 == 0:
                self.visualize_results(train_loader, epoch)
            
            # Save periodic checkpoint
            if (epoch + 1) % 20 == 0:
                self.save_checkpoint(epoch, g_loss, d_loss)

# ==================== Main Function ====================
def main():
    # Configuration
    config = Config()
    
    print("🎨 Animation Frame Prediction avec GAN + Vision Transformer")
    print(f"📊 Configuration:")
    print(f"  - Device: {config.DEVICE}")
    print(f"  - Batch Size: {config.BATCH_SIZE}")
    print(f"  - Image Size: {config.IMG_SIZE}")
    print(f"  - Sequence Length: {config.SEQUENCE_LENGTH}")
    
    # Dataset
    print("\n📁 Chargement du dataset...")
    train_dataset = EnhancedAnimeDataset(
        frames_dir=config.FRAMES_DIR,
        sequence_length=config.SEQUENCE_LENGTH,
        augment=True
    )
    
    val_dataset = EnhancedAnimeDataset(
        frames_dir=config.FRAMES_DIR,
        sequence_length=config.SEQUENCE_LENGTH,
        augment=False
    )
    
    # Split dataset
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(train_dataset)))
    
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    
    # DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"📊 Dataset: {len(train_subset)} train, {len(val_subset)} val")
    
    # Models
    print("\n🤖 Création des modèles...")
    generator = HybridGenerator(config)
    discriminator = PatchGANDiscriminator()
    
    # Count parameters
    g_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    d_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print(f"📊 Generator parameters: {g_params:,}")
    print(f"📊 Discriminator parameters: {d_params:,}")
    
    # Trainer
    trainer = GANTrainer(generator, discriminator, config)
    
    # Train
    trainer.train(train_loader, val_loader)
    
    print("\n✅ Entraînement terminé!")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(trainer.g_losses, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator Training')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(trainer.d_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Discriminator Training')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves_gan.png')
    plt.show()

# ==================== Utility Functions ====================
import torch.nn.functional as F

def extract_frames_from_video(video_path, output_dir, skip_frames=1, max_frames=None):
    """Extraire les frames d'une vidéo"""
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    
    print(f"📹 Extraction des frames de {video_path}...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % skip_frames == 0:
            output_path = os.path.join(output_dir, f'frame_{saved_count:04d}.png')
            cv2.imwrite(output_path, frame)
            saved_count += 1
            
            if max_frames and saved_count >= max_frames:
                break
        
        frame_count += 1
    
    cap.release()
    print(f"✅ {saved_count} frames extraites vers {output_dir}")

if __name__ == "__main__":
    # Option 1: Extraire les frames
    # extract_frames_from_video('anime_video.mp4', 'anime_frames', skip_frames=2, max_frames=1000)
    
    # Option 2: Entraîner le modèle
    main()