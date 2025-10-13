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
from tqdm import tqdm
import cv2
import random
from datetime import datetime
import gc
import json
from einops import rearrange
from einops.layers.torch import Rearrange

# ==================== Configuration Optimisée pour 12GB VRAM ====================
class Config:
    # Chemins
    FRAMES_DIR = "/kaggle/input/anima-s-dataset/test/"
    CHECKPOINT_DIR = "checkpoints"
    OUTPUT_DIR = "predictions"
    
    # Hyperparamètres OPTIMISÉS pour 12GB
    SEQUENCE_LENGTH = 3  # Réduit de 5 à 3
    IMG_SIZE = (128, 128)  # Réduit de 256x256 à 128x128
    BATCH_SIZE = 1  # Réduit de 4 à 1
    GRADIENT_ACCUMULATION_STEPS = 8  # Augmenté pour compenser le petit batch
    LEARNING_RATE_G = 0.0002
    LEARNING_RATE_D = 0.0001
    NUM_EPOCHS = 200
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model - Dimensions réduites
    HIDDEN_DIM = 256  # Réduit de 512 à 256
    NUM_LAYERS = 2  # Réduit de 3 à 2
    VIT_DIM = 384  # Réduit de 768 à 384
    VIT_DEPTH = 4  # Réduit de 12 à 4
    VIT_HEADS = 6  # Réduit de 12 à 6
    VIT_MLP_DIM = 1536  # Réduit de 3072 à 1536
    
    # Loss weights
    L1_WEIGHT = 100.0
    PERCEPTUAL_WEIGHT = 5.0  # Réduit
    GAN_WEIGHT = 1.0
    TEMPORAL_WEIGHT = 2.0  # Réduit
    
    # Training
    USE_MIXED_PRECISION = True  # Obligatoire pour économiser la mémoire
    CLIP_GRAD_NORM = 1.0
    GRADIENT_CHECKPOINTING = True  # Nouveau: pour économiser la mémoire

# ==================== Gestion Mémoire ====================
def clear_gpu_memory():
    """Libérer la mémoire GPU"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

# ==================== Lightweight Vision Transformer ====================
class LightweightAttention(nn.Module):
    """Attention légère avec moins de paramètres"""
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
    
    def forward(self, x):
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        # Utiliser torch.baddbmm pour économiser la mémoire
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class LightTransformerBlock(nn.Module):
    """Block Transformer léger"""
    def __init__(self, dim, heads, dim_head, mlp_dim):
        super().__init__()
        self.attn = LightweightAttention(dim, heads, dim_head)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# ==================== Lightweight Dataset ====================
class OptimizedAnimeDataset(Dataset):
    def __init__(self, frames_dir, sequence_length=3, img_size=(128, 128), augment=True):
        self.frames_dir = Path(frames_dir)
        self.sequence_length = sequence_length
        self.augment = augment
        
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        self.frame_paths = sorted(self.frames_dir.glob("frame_*.png"))
        self.num_sequences = len(self.frame_paths) - sequence_length
        
        if self.num_sequences <= 0:
            raise ValueError(f"Pas assez de frames. Minimum: {sequence_length + 1}")
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        input_frames = []
        
        for i in range(self.sequence_length):
            frame_path = self.frame_paths[idx + i]
            frame = Image.open(frame_path).convert('RGB')
            
            # Augmentation simple et légère
            if self.augment and random.random() > 0.5:
                frame = transforms.functional.hflip(frame)
            
            frame = self.transform(frame)
            input_frames.append(frame)
        
        target_frame_path = self.frame_paths[idx + self.sequence_length]
        target_frame = Image.open(target_frame_path).convert('RGB')
        
        if self.augment and random.random() > 0.5:
            target_frame = transforms.functional.hflip(target_frame)
        
        target_frame = self.transform(target_frame)
        
        return torch.stack(input_frames), target_frame

# ==================== Lightweight ConvLSTM ====================
class LightConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size,
            padding=padding
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

# ==================== Lightweight Generator ====================
class LightweightGenerator(nn.Module):
    """Générateur optimisé pour faible VRAM"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Encoder léger
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),  # 64->32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1),  # 128->64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),  # 256->128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),  # 512->256
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        
        # ConvLSTM léger
        self.convlstm = LightConvLSTMCell(256, 256)
        
        # Mini Vision Transformer
        self.use_vit = False  # Désactiver pour économiser la mémoire
        if self.use_vit:
            self.patch_size = 8
            self.num_patches = (config.IMG_SIZE[0] // 8) * (config.IMG_SIZE[1] // 8)
            
            self.patch_embed = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                         p1=self.patch_size, p2=self.patch_size),
                nn.Linear(self.patch_size * self.patch_size * 256, config.VIT_DIM)
            )
            
            self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, config.VIT_DIM))
            
            self.transformer_blocks = nn.ModuleList([
                LightTransformerBlock(
                    config.VIT_DIM, 
                    config.VIT_HEADS, 
                    config.VIT_DIM // config.VIT_HEADS,
                    config.VIT_MLP_DIM
                ) for _ in range(2)  # Seulement 2 blocks
            ])
            
            self.norm = nn.LayerNorm(config.VIT_DIM)
            
            self.unpatch = nn.Sequential(
                nn.Linear(config.VIT_DIM, self.patch_size * self.patch_size * 256),
                Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                         h=config.IMG_SIZE[0]//self.patch_size,
                         w=config.IMG_SIZE[1]//self.patch_size,
                         p1=self.patch_size, p2=self.patch_size)
            )
        
        # Decoder léger
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, 1, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        
        # Initialiser les états LSTM
        h_lstm = torch.zeros(batch_size, 256, h//16, w//16).to(x.device)
        c_lstm = torch.zeros_like(h_lstm)
        
        # Process chaque frame
        for t in range(seq_len):
            encoded = self.encoder(x[:, t])
            h_lstm, c_lstm = self.convlstm(encoded, (h_lstm, c_lstm))
        
        # Utiliser la dernière sortie
        features = h_lstm
        
        # Vision Transformer (si activé)
        if self.use_vit:
            b, c, h_feat, w_feat = features.shape
            
            # Ajuster la taille si nécessaire
            if h_feat * w_feat != self.num_patches:
                features = torch.nn.functional.interpolate(
                    features, size=(8, 8), mode='bilinear', align_corners=False
                )
            
            # Patches
            patches = self.patch_embed(features)
            patches = patches + self.pos_embedding
            
            # Transformer blocks
            for transformer in self.transformer_blocks:
                patches = transformer(patches)
            
            patches = self.norm(patches)
            features = self.unpatch(patches)
        
        # Decoder
        output = self.decoder(features)
        
        return output

# ==================== Lightweight Discriminator ====================
class LightweightDiscriminator(nn.Module):
    """Discriminateur léger"""
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(6, 32, 4, 2, 1),  # 64->32
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1),  # 128->64
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),  # 256->128
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 1, 1),  # 512->256
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 1, 1)
        )
    
    def forward(self, input_frame, target):
        # Prendre la dernière frame si séquence
        if len(input_frame.shape) == 5:
            input_frame = input_frame[:, -1]
        
        combined = torch.cat([input_frame, target], dim=1)
        return self.model(combined)

# ==================== Simple Perceptual Loss ====================
class SimplePerceptualLoss(nn.Module):
    """Version simplifiée de la loss perceptuelle"""
    def __init__(self, device):
        super().__init__()
        # Utiliser seulement les premières couches de VGG pour économiser la mémoire
        vgg = models.vgg16(pretrained=True).features[:16].to(device).eval()
        self.features = vgg
        
        for p in self.parameters():
            p.requires_grad = False
    
    def forward(self, x, y):
        # Normaliser pour VGG
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        
        x = (x + 1) / 2  # Dénormaliser de [-1, 1] à [0, 1]
        y = (y + 1) / 2
        
        x = (x - mean) / std
        y = (y - mean) / std
        
        x_features = self.features(x)
        y_features = self.features(y)
        
        return torch.nn.functional.l1_loss(x_features, y_features)

# ==================== Memory-Efficient Trainer ====================
class MemoryEfficientTrainer:
    def __init__(self, generator, discriminator, config):
        self.generator = generator.to(config.DEVICE)
        self.discriminator = discriminator.to(config.DEVICE)
        self.config = config
        
        # Optimizers avec moins de mémoire
        self.optimizer_G = optim.AdamW(
            generator.parameters(), 
            lr=config.LEARNING_RATE_G,
            weight_decay=0.01
        )
        self.optimizer_D = optim.AdamW(
            discriminator.parameters(), 
            lr=config.LEARNING_RATE_D,
            weight_decay=0.01
        )
        
        # Losses
        self.criterion_GAN = nn.BCEWithLogitsLoss()
        self.criterion_L1 = nn.L1Loss()
        self.perceptual_loss = SimplePerceptualLoss(config.DEVICE)
        
        # Mixed precision (obligatoire)
        self.scaler_G = GradScaler()
        self.scaler_D = GradScaler()
        
        # Schedulers
        self.scheduler_G = optim.lr_scheduler.OneCycleLR(
            self.optimizer_G, 
            max_lr=config.LEARNING_RATE_G,
            total_steps=config.NUM_EPOCHS * 100,  # Estimation
            pct_start=0.1
        )
        self.scheduler_D = optim.lr_scheduler.OneCycleLR(
            self.optimizer_D,
            max_lr=config.LEARNING_RATE_D,
            total_steps=config.NUM_EPOCHS * 100,
            pct_start=0.1
        )
        
        # Metrics
        self.metrics = {
            'g_losses': [],
            'd_losses': [],
            'val_losses': []
        }
        
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    def train_discriminator(self, real_batch, input_frames):
        self.optimizer_D.zero_grad()
        
        with autocast():
            # Générer les fakes
            with torch.no_grad():
                fake_batch = self.generator(input_frames)
            
            # Real
            real_pred = self.discriminator(input_frames, real_batch)
            real_labels = torch.ones_like(real_pred) * 0.9
            real_loss = self.criterion_GAN(real_pred, real_labels)
            
            # Fake
            fake_pred = self.discriminator(input_frames, fake_batch)
            fake_labels = torch.zeros_like(fake_pred) + 0.1
            fake_loss = self.criterion_GAN(fake_pred, fake_labels)
            
            d_loss = (real_loss + fake_loss) / 2
        
        self.scaler_D.scale(d_loss).backward()
        self.scaler_D.step(self.optimizer_D)
        self.scaler_D.update()
        
        return d_loss.item()
    
    def train_generator(self, input_frames, target_frame):
        self.optimizer_G.zero_grad()
        
        with autocast():
            # Generate
            generated_frame = self.generator(input_frames)
            
            # GAN Loss
            fake_pred = self.discriminator(input_frames, generated_frame)
            real_labels = torch.ones_like(fake_pred)
            gan_loss = self.criterion_GAN(fake_pred, real_labels)
            
            # L1 Loss
            l1_loss = self.criterion_L1(generated_frame, target_frame)
            
            # Perceptual Loss (optionnel pour économiser la mémoire)
            if self.config.PERCEPTUAL_WEIGHT > 0:
                perc_loss = self.perceptual_loss(generated_frame, target_frame)
            else:
                perc_loss = torch.tensor(0.0).to(self.config.DEVICE)
            
            # Total
            g_loss = (
                self.config.GAN_WEIGHT * gan_loss +
                self.config.L1_WEIGHT * l1_loss +
                self.config.PERCEPTUAL_WEIGHT * perc_loss
            )
        
        self.scaler_G.scale(g_loss).backward()
        
        # Gradient clipping
        self.scaler_G.unscale_(self.optimizer_G)
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.config.CLIP_GRAD_NORM)
        
        self.scaler_G.step(self.optimizer_G)
        self.scaler_G.update()
        
        return {
            'total': g_loss.item(),
            'gan': gan_loss.item(),
            'l1': l1_loss.item(),
            'perceptual': perc_loss.item() if self.config.PERCEPTUAL_WEIGHT > 0 else 0
        }
    
    def train_epoch(self, dataloader, epoch):
        self.generator.train()
        self.discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}") as pbar:
            for batch_idx, (input_frames, target_frame) in enumerate(pbar):
                input_frames = input_frames.to(self.config.DEVICE)
                target_frame = target_frame.to(self.config.DEVICE)
                
                # Train D every 2 iterations
                if batch_idx % 2 == 0:
                    d_loss = self.train_discriminator(target_frame, input_frames)
                    epoch_d_loss += d_loss
                
                # Train G
                g_losses = self.train_generator(input_frames, target_frame)
                epoch_g_loss += g_losses['total']
                
                # Update schedulers
                self.scheduler_G.step()
                self.scheduler_D.step()
                
                # Update progress
                pbar.set_postfix({
                    'G': f"{g_losses['total']:.4f}",
                    'D': f"{d_loss:.4f}" if batch_idx % 2 == 0 else "N/A",
                    'L1': f"{g_losses['l1']:.4f}"
                })
                
                # Libérer la mémoire périodiquement
                if batch_idx % 10 == 0:
                    clear_gpu_memory()
        
        return epoch_g_loss / len(dataloader), epoch_d_loss / max(1, len(dataloader) // 2)
    
    def validate(self, dataloader):
        self.generator.eval()
        val_loss = 0
        
        with torch.no_grad():
            for input_frames, target_frame in dataloader:
                input_frames = input_frames.to(self.config.DEVICE)
                target_frame = target_frame.to(self.config.DEVICE)
                
                with autocast():
                    generated_frame = self.generator(input_frames)
                    loss = self.criterion_L1(generated_frame, target_frame)
                
                val_loss += loss.item()
                
                # Libérer la mémoire
                del input_frames, target_frame, generated_frame
                clear_gpu_memory()
        
        return val_loss / len(dataloader)
    
    def save_checkpoint(self, epoch, g_loss, d_loss):
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_loss': g_loss,
            'd_loss': d_loss
        }
        
        path = os.path.join(
            self.config.CHECKPOINT_DIR,
            f'checkpoint_epoch_{epoch}.pth'
        )
        torch.save(checkpoint, path)
        print(f"✅ Checkpoint: {path}")
    
    def visualize_results(self, dataloader, epoch):
        self.generator.eval()
        
        with torch.no_grad():
            data = next(iter(dataloader))
            input_frames = data[0][:1].to(self.config.DEVICE)  # Prendre seulement 1 sample
            target_frame = data[1][:1].to(self.config.DEVICE)
            
            with autocast():
                generated_frame = self.generator(input_frames)
            
            # Sauvegarder avec OpenCV
            def tensor_to_image(tensor):
                img = tensor[0].cpu().numpy()
                img = (img + 1) / 2
                img = np.transpose(img, (1, 2, 0))
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Créer grille
            input_img = tensor_to_image(input_frames[:, -1])
            target_img = tensor_to_image(target_frame)
            generated_img = tensor_to_image(generated_frame)
            
            # Combiner horizontalement
            grid = np.hstack([input_img, target_img, generated_img])
            
            save_path = os.path.join(
                self.config.OUTPUT_DIR,
                f'epoch_{epoch+1}.png'
            )
            cv2.imwrite(save_path, grid)
            print(f"📸 Visualisation: {save_path}")
            
            # Libérer mémoire
            del input_frames, target_frame, generated_frame
            clear_gpu_memory()
    
    def train(self, train_loader, val_loader=None):
        print(f"🚀 Training on {self.config.DEVICE}")
        print(f"📊 Batch Size: {self.config.BATCH_SIZE}")
        print(f"📊 Image Size: {self.config.IMG_SIZE}")
        print(f"📊 Mixed Precision: Enabled")
        print(f"📊 Memory Optimization: Enabled")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\n{'='*50}")
            print(f"📅 Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
            
            # Training
            g_loss, d_loss = self.train_epoch(train_loader, epoch)
            self.metrics['g_losses'].append(g_loss)
            self.metrics['d_losses'].append(d_loss)
            
            print(f"Generator Loss: {g_loss:.4f}")
            print(f"Discriminator Loss: {d_loss:.4f}")
            
            # Validation
            if val_loader and (epoch + 1) % 5 == 0:
                val_loss = self.validate(val_loader)
                self.metrics['val_losses'].append(val_loss)
                print(f"Validation Loss: {val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(epoch, g_loss, d_loss)
            
            # Visualize
            if (epoch + 1) % 5 == 0:
                self.visualize_results(train_loader, epoch)
            
            # Save metrics
            if (epoch + 1) % 10 == 0:
                with open(os.path.join(self.config.OUTPUT_DIR, 'metrics.json'), 'w') as f:
                    json.dump(self.metrics, f, indent=2)
            
            # Libérer mémoire
            clear_gpu_memory()

# ==================== Main ====================
def main():
    # Configuration
    config = Config()
    
    # Définir les variables d'environnement pour optimiser CUDA
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    print("🎨 Anime Frame Prediction (Optimized for 12GB VRAM)")
    print(f"📊 Device: {config.DEVICE}")
    
    # Clear GPU memory before starting
    clear_gpu_memory()
    
    # Dataset
    print("\n📁 Loading dataset...")
    dataset = OptimizedAnimeDataset(
        frames_dir=config.FRAMES_DIR,
        sequence_length=config.SEQUENCE_LENGTH,
        img_size=config.IMG_SIZE,
        augment=True
    )
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # DataLoaders avec pin_memory=False pour économiser la RAM
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # 0 pour économiser la mémoire
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Models
    print("\n🤖 Creating models...")
    generator = LightweightGenerator(config)
    discriminator = LightweightDiscriminator()
    
    # Count parameters
    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"📊 Generator: {g_params:,} parameters")
    print(f"📊 Discriminator: {d_params:,} parameters")
    
    # Calculate memory usage
    param_memory = (g_params + d_params) * 4 / (1024**3)  # in GB
    print(f"📊 Model parameters memory: ~{param_memory:.2f} GB")
    
    # Trainer
    trainer = MemoryEfficientTrainer(generator, discriminator, config)
    
    # Train
    trainer.train(train_loader, val_loader)
    
    print("\n✅ Training complete!")

def extract_frames(video_path, output_dir, max_frames=500):
    """Extract frames with limited number for testing"""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    count = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize to save disk space
        frame = cv2.resize(frame, (256, 256))
        cv2.imwrite(f"{output_dir}/frame_{count:04d}.png", frame)
        count += 1
    
    cap.release()
    print(f"✅ Extracted {count} frames")

if __name__ == "__main__":
    main()