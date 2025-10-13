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
import gc
import json
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn.functional as F

# ==================== Configuration pour Haute Qualité ====================
class Config:
    # Chemins
    FRAMES_DIR = "/kaggle/input/anima-s-dataset/test/"
    CHECKPOINT_DIR = "checkpoints"
    OUTPUT_DIR = "predictions"
    
    # Hyperparamètres - Optimisés pour qualité
    SEQUENCE_LENGTH = 3
    IMG_SIZE = (256, 256)  # On augmente pour plus de détails
    PATCH_SIZE = (64, 64)  # Taille des patches pour l'entraînement progressif
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 8
    LEARNING_RATE_G = 0.0001
    LEARNING_RATE_D = 0.00005
    LEARNING_RATE_R = 0.0001  # Pour le refinement network
    NUM_EPOCHS = 300
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Architecture
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    
    # Loss weights - Ajustés pour la qualité
    RECONSTRUCTION_WEIGHT = 50.0  # Nouveau : pour la reconstruction parfaite
    L1_WEIGHT = 100.0
    PERCEPTUAL_WEIGHT = 20.0  # Augmenté pour plus de détails
    TEXTURE_WEIGHT = 10.0  # Nouveau : pour les détails fins
    EDGE_WEIGHT = 5.0  # Nouveau : pour les contours nets
    GAN_WEIGHT = 1.0
    SSIM_WEIGHT = 10.0  # Nouveau : structural similarity
    
    # Training
    USE_MIXED_PRECISION = True
    CLIP_GRAD_NORM = 1.0
    USE_PROGRESSIVE_TRAINING = True  # Entraînement progressif pour la qualité

# ==================== Module de Super-Résolution ====================
class ResidualBlock(nn.Module):
    """Block résiduel pour la super-résolution"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class QualityRefinementModule(nn.Module):
    """Module pour raffiner la qualité des images générées"""
    def __init__(self, in_channels=3, num_blocks=8):
        super().__init__()
        
        # Extraction de features multi-échelle
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, 9, padding=4),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks pour capturer les détails
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_blocks)])
        
        # Reconstruction haute qualité
        self.reconstruction = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_channels, 9, padding=4),
            nn.Tanh()
        )
    
    def forward(self, x):
        initial_features = self.initial(x)
        res_features = self.res_blocks(initial_features)
        refined = self.reconstruction(res_features + initial_features)
        return refined

# ==================== Générateur avec Auto-Encoder et Raffinement ====================
class HighQualityGenerator(nn.Module):
    """Générateur avec reconstruction haute qualité et prédiction"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # ========== ENCODER (Partagé) ==========
        self.encoder = nn.ModuleList([
            # Level 1: 256 -> 128
            nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),
                ResidualBlock(64),  # Détails haute fréquence
            ),
            # Level 2: 128 -> 64
            nn.Sequential(
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                ResidualBlock(128),
            ),
            # Level 3: 64 -> 32
            nn.Sequential(
                nn.Conv2d(128, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                ResidualBlock(256),
            ),
            # Level 4: 32 -> 16
            nn.Sequential(
                nn.Conv2d(256, 512, 4, 2, 1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2),
                ResidualBlock(512),
            )
        ])
        
        # ========== TEMPORAL MODULE ==========
        self.temporal_module = nn.LSTM(
            input_size=512 * 16 * 16,  # Flatten features
            hidden_size=2048,
            num_layers=2,
            batch_first=True
        )
        
        # ========== RECONSTRUCTION DECODER ==========
        # Pour reconstruire parfaitement l'image d'entrée
        self.reconstruction_decoder = nn.ModuleList([
            # Level 4: 16 -> 32
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                ResidualBlock(256),
            ),
            # Level 3: 32 -> 64
            nn.Sequential(
                nn.ConvTranspose2d(256 + 256, 128, 4, 2, 1),  # +256 pour skip connection
                nn.BatchNorm2d(128),
                nn.ReLU(),
                ResidualBlock(128),
            ),
            # Level 2: 64 -> 128
            nn.Sequential(
                nn.ConvTranspose2d(128 + 128, 64, 4, 2, 1),  # +128 pour skip
                nn.BatchNorm2d(64),
                nn.ReLU(),
                ResidualBlock(64),
            ),
            # Level 1: 128 -> 256
            nn.Sequential(
                nn.ConvTranspose2d(64 + 64, 32, 4, 2, 1),  # +64 pour skip
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 3, 3, 1, 1),
                nn.Tanh()
            )
        ])
        
        # ========== PREDICTION DECODER ==========
        # Pour prédire la frame suivante
        self.prediction_decoder = nn.ModuleList([
            # Même architecture que reconstruction_decoder
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                ResidualBlock(256),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                ResidualBlock(128),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                ResidualBlock(64),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 3, 3, 1, 1),
                nn.Tanh()
            )
        ])
        
        # ========== QUALITY REFINEMENT ==========
        self.quality_refiner = QualityRefinementModule(in_channels=3, num_blocks=6)
        
        # ========== DETAIL ENHANCEMENT ==========
        self.detail_enhancer = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),  # 6 channels: original + refined
            nn.ReLU(),
            ResidualBlock(32),
            ResidualBlock(32),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh()
        )
    
    def encode_frame(self, frame):
        """Encoder une frame avec skip connections"""
        skip_connections = []
        x = frame
        
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            skip_connections.append(x)
        
        return x, skip_connections
    
    def decode_reconstruction(self, features, skip_connections):
        """Décoder pour reconstruire l'image originale"""
        x = features
        
        # Premier décodage sans skip
        x = self.reconstruction_decoder[0](x)
        
        # Décodages suivants avec skip connections
        for i in range(1, len(self.reconstruction_decoder)):
            if i < len(skip_connections):
                # Ajouter la skip connection
                skip = skip_connections[-(i+1)]
                x = torch.cat([x, skip], dim=1)
            x = self.reconstruction_decoder[i](x)
        
        return x
    
    def decode_prediction(self, features):
        """Décoder pour prédire la frame suivante"""
        x = features
        
        for decoder_layer in self.prediction_decoder:
            x = decoder_layer(x)
        
        return x
    
    def forward(self, x, return_all=False):
        batch_size, seq_len, c, h, w = x.size()
        
        # ========== PHASE 1: ENCODER + RECONSTRUCTION ==========
        encoded_features = []
        reconstructions = []
        all_skip_connections = []
        
        for t in range(seq_len):
            features, skip_connections = self.encode_frame(x[:, t])
            encoded_features.append(features)
            all_skip_connections.append(skip_connections)
            
            # Reconstruire la frame actuelle pour vérifier la qualité
            if return_all:
                reconstruction = self.decode_reconstruction(features, skip_connections)
                reconstruction = self.quality_refiner(reconstruction)
                reconstructions.append(reconstruction)
        
        # ========== PHASE 2: TEMPORAL PROCESSING ==========
        # Flatten et process temporel
        flattened_features = []
        for feat in encoded_features:
            flattened = feat.view(batch_size, -1)
            flattened_features.append(flattened)
        
        temporal_input = torch.stack(flattened_features, dim=1)
        temporal_output, _ = self.temporal_module(temporal_input)
        
        # Prendre la dernière sortie temporelle
        last_temporal = temporal_output[:, -1]
        last_temporal = last_temporal.view(batch_size, 512, 16, 16)
        
        # ========== PHASE 3: PREDICTION ==========
        # Prédire la frame suivante
        predicted_frame = self.decode_prediction(last_temporal)
        
        # ========== PHASE 4: QUALITY REFINEMENT ==========
        # Raffiner la prédiction
        refined_prediction = self.quality_refiner(predicted_frame)
        
        # ========== PHASE 5: DETAIL ENHANCEMENT ==========
        # Combiner avec la dernière frame pour améliorer les détails
        last_input_frame = x[:, -1]
        combined = torch.cat([refined_prediction, last_input_frame], dim=1)
        final_prediction = self.detail_enhancer(combined)
        
        # Fusion finale avec résiduel
        final_prediction = 0.7 * refined_prediction + 0.3 * final_prediction
        
        if return_all:
            return final_prediction, reconstructions
        return final_prediction

# ==================== Discriminateur Multi-Échelle ====================
class MultiScaleDiscriminator(nn.Module):
    """Discriminateur multi-échelle pour juger la qualité à différents niveaux"""
    def __init__(self):
        super().__init__()
        
        # Discriminateur pour détails fins (haute résolution)
        self.fine_discriminator = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 1)
        )
        
        # Discriminateur pour structure globale (basse résolution)
        self.coarse_discriminator = nn.Sequential(
            nn.AvgPool2d(2),  # Downsample
            nn.Conv2d(6, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 4, 1, 1)
        )
    
    def forward(self, input_frame, target):
        if len(input_frame.shape) == 5:
            input_frame = input_frame[:, -1]
        
        combined = torch.cat([input_frame, target], dim=1)
        
        fine_output = self.fine_discriminator(combined)
        coarse_output = self.coarse_discriminator(combined)
        
        return [fine_output, coarse_output]

# ==================== Loss Functions Avancées ====================
class EdgeLoss(nn.Module):
    """Loss pour préserver les contours nets"""
    def __init__(self):
        super().__init__()
        # Sobel filters
        self.sobel_x = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        
        sobel_x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.sobel_x.weight = nn.Parameter(sobel_x_kernel, requires_grad=False)
        self.sobel_y.weight = nn.Parameter(sobel_y_kernel, requires_grad=False)
    
    def forward(self, pred, target):
        # Convert to grayscale
        pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
        
        # Compute edges
        pred_edge_x = self.sobel_x(pred_gray)
        pred_edge_y = self.sobel_y(pred_gray)
        pred_edges = torch.sqrt(pred_edge_x**2 + pred_edge_y**2)
        
        target_edge_x = self.sobel_x(target_gray)
        target_edge_y = self.sobel_y(target_gray)
        target_edges = torch.sqrt(target_edge_x**2 + target_edge_y**2)
        
        return F.l1_loss(pred_edges, target_edges)

class TextureLoss(nn.Module):
    """Loss pour préserver les textures et détails fins"""
    def __init__(self, device):
        super().__init__()
        # Utiliser VGG pour extraire les features de texture
        vgg = models.vgg19(pretrained=True).features
        self.layers = nn.ModuleList([
            vgg[:4],   # Conv1_2
            vgg[4:9],  # Conv2_2
            vgg[9:14]  # Conv3_2
        ]).to(device).eval()
        
        for p in self.parameters():
            p.requires_grad = False
    
    def gram_matrix(self, x):
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
    
    def forward(self, pred, target):
        loss = 0
        
        # Normaliser pour VGG
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        
        pred = (pred + 1) / 2
        target = (target + 1) / 2
        pred = (pred - mean) / std
        target = (target - mean) / std
        
        for layer in self.layers:
            pred = layer(pred)
            target = layer(target)
            
            pred_gram = self.gram_matrix(pred)
            target_gram = self.gram_matrix(target)
            
            loss += F.l1_loss(pred_gram, target_gram)
        
        return loss

class SSIMLoss(nn.Module):
    """Structural Similarity Index Loss"""
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = self.create_window(window_size, self.channel)
    
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
    
    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, img1, img2):
        channel = img1.size()[1]
        
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)

# ==================== Trainer Haute Qualité ====================
class HighQualityTrainer:
    def __init__(self, generator, discriminator, config):
        self.generator = generator.to(config.DEVICE)
        self.discriminator = discriminator.to(config.DEVICE)
        self.config = config
        
        # Optimizers
        self.optimizer_G = optim.AdamW(
            generator.parameters(),
            lr=config.LEARNING_RATE_G,
            betas=(0.5, 0.999),
            weight_decay=0.01
        )
        self.optimizer_D = optim.AdamW(
            discriminator.parameters(),
            lr=config.LEARNING_RATE_D,
            betas=(0.5, 0.999),
            weight_decay=0.01
        )
        
        # Losses
        self.criterion_GAN = nn.BCEWithLogitsLoss()
        self.criterion_L1 = nn.L1Loss()
        self.criterion_L2 = nn.MSELoss()
        
        # Advanced losses
        from torchvision.models import vgg19
        self.perceptual_loss = self.setup_perceptual_loss()
        self.edge_loss = EdgeLoss()
        self.texture_loss = TextureLoss(config.DEVICE)
        self.ssim_loss = SSIMLoss()
        
        # Mixed precision
        self.scaler_G = GradScaler()
        self.scaler_D = GradScaler()
        
        # Metrics
        self.metrics = {
            'g_losses': [],
            'd_losses': [],
            'reconstruction_losses': [],
            'prediction_losses': [],
            'quality_scores': []
        }
        
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    def setup_perceptual_loss(self):
        """Setup VGG perceptual loss"""
        vgg = models.vgg19(pretrained=True).features[:23].to(self.config.DEVICE).eval()
        for p in vgg.parameters():
            p.requires_grad = False
        return vgg
    
    def compute_perceptual_loss(self, pred, target):
        """Compute perceptual loss using VGG features"""
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        
        pred = (pred + 1) / 2
        target = (target + 1) / 2
        pred = (pred - mean) / std
        target = (target - mean) / std
        
        pred_features = self.perceptual_loss(pred)
        target_features = self.perceptual_loss(target)
        
        return F.l1_loss(pred_features, target_features)
    
    def train_discriminator(self, real_batch, input_frames):
        self.optimizer_D.zero_grad()
        
        with autocast():
            # Generate fake
            with torch.no_grad():
                fake_batch, _ = self.generator(input_frames, return_all=True)
            
            # Get multi-scale outputs
            real_outputs = self.discriminator(input_frames, real_batch)
            fake_outputs = self.discriminator(input_frames, fake_batch)
            
            d_loss = 0
            for real_out, fake_out in zip(real_outputs, fake_outputs):
                real_labels = torch.ones_like(real_out) * 0.9
                fake_labels = torch.zeros_like(fake_out) + 0.1
                
                d_loss += self.criterion_GAN(real_out, real_labels)
                d_loss += self.criterion_GAN(fake_out, fake_labels)
            
            d_loss = d_loss / (2 * len(real_outputs))
        
        self.scaler_D.scale(d_loss).backward()
        self.scaler_D.step(self.optimizer_D)
        self.scaler_D.update()
        
        return d_loss.item()
    
    def train_generator(self, input_frames, target_frame):
        self.optimizer_G.zero_grad()
        
        with autocast():
            # Generate with reconstructions
            predicted_frame, reconstructions = self.generator(input_frames, return_all=True)
            
            # ========== RECONSTRUCTION LOSS ==========
            # Comparer les reconstructions avec les frames originales
            reconstruction_loss = 0
            for t in range(len(reconstructions)):
                original_frame = input_frames[:, t]
                reconstructed_frame = reconstructions[t]
                
                # L1 + L2 pour reconstruction parfaite
                reconstruction_loss += self.criterion_L1(reconstructed_frame, original_frame)
                reconstruction_loss += 0.5 * self.criterion_L2(reconstructed_frame, original_frame)
                
                # SSIM pour la structure
                reconstruction_loss += self.config.SSIM_WEIGHT * self.ssim_loss(
                    reconstructed_frame, original_frame
                )
            
            reconstruction_loss = reconstruction_loss / len(reconstructions)
            
            # ========== PREDICTION LOSSES ==========
            # L1 Loss
            l1_loss = self.criterion_L1(predicted_frame, target_frame)
            
            # Perceptual Loss
            perceptual_loss = self.compute_perceptual_loss(predicted_frame, target_frame)
            
            # Edge Loss (pour les contours nets)
            edge_loss = self.edge_loss(predicted_frame, target_frame)
            
            # Texture Loss (pour les détails)
            texture_loss = self.texture_loss(predicted_frame, target_frame)
            
            # SSIM Loss
            ssim_loss = self.ssim_loss(predicted_frame, target_frame)
            
            # ========== GAN LOSS ==========
            fake_outputs = self.discriminator(input_frames, predicted_frame)
            gan_loss = 0
            for fake_out in fake_outputs:
                real_labels = torch.ones_like(fake_out)
                gan_loss += self.criterion_GAN(fake_out, real_labels)
            gan_loss = gan_loss / len(fake_outputs)
            
            # ========== TOTAL LOSS ==========
            g_loss = (
                self.config.RECONSTRUCTION_WEIGHT * reconstruction_loss +
                self.config.L1_WEIGHT * l1_loss +
                self.config.PERCEPTUAL_WEIGHT * perceptual_loss +
                self.config.EDGE_WEIGHT * edge_loss +
                self.config.TEXTURE_WEIGHT * texture_loss +
                self.config.SSIM_WEIGHT * ssim_loss +
                self.config.GAN_WEIGHT * gan_loss
            )
        
        self.scaler_G.scale(g_loss).backward()
        
        # Gradient clipping
        self.scaler_G.unscale_(self.optimizer_G)
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.config.CLIP_GRAD_NORM)
        
        self.scaler_G.step(self.optimizer_G)
        self.scaler_G.update()
        
        return {
            'total': g_loss.item(),
            'reconstruction': reconstruction_loss.item(),
            'l1': l1_loss.item(),
            'perceptual': perceptual_loss.item(),
            'edge': edge_loss.item(),
            'texture': texture_loss.item(),
            'ssim': ssim_loss.item(),
            'gan': gan_loss.item()
        }
    
    def train_epoch(self, dataloader, epoch):
        self.generator.train()
        self.discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_reconstruction_loss = 0
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}") as pbar:
            for batch_idx, (input_frames, target_frame) in enumerate(pbar):
                input_frames = input_frames.to(self.config.DEVICE)
                target_frame = target_frame.to(self.config.DEVICE)
                
                # Train Discriminator
                if batch_idx % 2 == 0:
                    d_loss = self.train_discriminator(target_frame, input_frames)
                    epoch_d_loss += d_loss
                
                # Train Generator
                g_losses = self.train_generator(input_frames, target_frame)
                epoch_g_loss += g_losses['total']
                epoch_reconstruction_loss += g_losses['reconstruction']
                
                # Update progress
                pbar.set_postfix({
                    'G': f"{g_losses['total']:.3f}",
                    'Recon': f"{g_losses['reconstruction']:.3f}",
                    'L1': f"{g_losses['l1']:.3f}",
                    'Edge': f"{g_losses['edge']:.3f}",
                    'SSIM': f"{g_losses['ssim']:.3f}"
                })
                
                # Clear memory periodically
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
        
        return (
            epoch_g_loss / len(dataloader),
            epoch_d_loss / max(1, len(dataloader) // 2),
            epoch_reconstruction_loss / len(dataloader)
        )
    
    def visualize_results(self, dataloader, epoch):
        self.generator.eval()
        
        with torch.no_grad():
            data = next(iter(dataloader))
            input_frames = data[0][:1].to(self.config.DEVICE)
            target_frame = data[1][:1].to(self.config.DEVICE)
            
            with autocast():
                predicted_frame, reconstructions = self.generator(input_frames, return_all=True)
            
            # Helper function
            def tensor_to_image(tensor):
                img = tensor[0].cpu().numpy()
                img = (img + 1) / 2
                img = np.transpose(img, (1, 2, 0))
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Create comparison grid
            images = []
            
            # Original input
            input_img = tensor_to_image(input_frames[:, -1])
            images.append(input_img)
            
            # Reconstruction of input
            if reconstructions:
                recon_img = tensor_to_image(reconstructions[-1])
                images.append(recon_img)
            
            # Target
            target_img = tensor_to_image(target_frame)
            images.append(target_img)
            
            # Prediction
            pred_img = tensor_to_image(predicted_frame)
            images.append(pred_img)
            
            # Stack images
            top_row = np.hstack(images[:2]) if len(images) > 2 else images[0]
            bottom_row = np.hstack(images[2:]) if len(images) > 2 else images[1]
            grid = np.vstack([top_row, bottom_row])
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(grid, "Input", (10, 30), font, 1, (255, 255, 255), 2)
            cv2.putText(grid, "Reconstruction", (266, 30), font, 1, (255, 255, 255), 2)
            cv2.putText(grid, "Target", (10, 286), font, 1, (255, 255, 255), 2)
            cv2.putText(grid, "Prediction", (266, 286), font, 1, (255, 255, 255), 2)
            
            save_path = os.path.join(self.config.OUTPUT_DIR, f'epoch_{epoch+1}.png')
            cv2.imwrite(save_path, grid)
            print(f"📸 Visualisation: {save_path}")
            
            # Clear memory
            torch.cuda.empty_cache()
    
    def train(self, train_loader, val_loader=None):
        print(f"🚀 High Quality Training on {self.config.DEVICE}")
        print(f"📊 Focus on: Reconstruction + Prediction Quality")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\n{'='*60}")
            print(f"📅 Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
            
            # Training
            g_loss, d_loss, recon_loss = self.train_epoch(train_loader, epoch)
            
            self.metrics['g_losses'].append(g_loss)
            self.metrics['d_losses'].append(d_loss)
            self.metrics['reconstruction_losses'].append(recon_loss)
            
            print(f"📊 Generator Loss: {g_loss:.4f}")
            print(f"📊 Discriminator Loss: {d_loss:.4f}")
            print(f"📊 Reconstruction Loss: {recon_loss:.4f}")
            
            # Visualize
            if (epoch + 1) % 5 == 0:
                self.visualize_results(train_loader, epoch)
            
            # Save checkpoint
            if (epoch + 1) % 20 == 0:
                self.save_checkpoint(epoch, g_loss, d_loss)
            
            # Save metrics
            if (epoch + 1) % 10 == 0:
                with open(os.path.join(self.config.OUTPUT_DIR, 'metrics.json'), 'w') as f:
                    json.dump(self.metrics, f, indent=2)
    
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
            f'checkpoint_epoch_{epoch}_quality.pth'
        )
        torch.save(checkpoint, path)
        print(f"✅ Checkpoint saved: {path}")

# ==================== Dataset ====================
class QualityAnimeDataset(Dataset):
    def __init__(self, frames_dir, sequence_length=3, img_size=(256, 256)):
        self.frames_dir = Path(frames_dir)
        self.sequence_length = sequence_length
        
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        self.frame_paths = sorted(self.frames_dir.glob("frame_*.png"))
        self.num_sequences = len(self.frame_paths) - sequence_length
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        input_frames = []
        
        for i in range(self.sequence_length):
            frame = Image.open(self.frame_paths[idx + i]).convert('RGB')
            frame = self.transform(frame)
            input_frames.append(frame)
        
        target_frame = Image.open(self.frame_paths[idx + self.sequence_length]).convert('RGB')
        target_frame = self.transform(target_frame)
        
        return torch.stack(input_frames), target_frame

# ==================== Main ====================
def main():
    config = Config()
    
    # Set CUDA optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.cuda.empty_cache()
    
    print("🎨 High Quality Anime Frame Prediction")
    print("✨ Focus: Reconstruction Quality + Prediction Accuracy")
    print(f"📊 Device: {config.DEVICE}")
    
    # Dataset
    print("\n📁 Loading dataset...")
    dataset = QualityAnimeDataset(
        frames_dir=config.FRAMES_DIR,
        sequence_length=config.SEQUENCE_LENGTH,
        img_size=config.IMG_SIZE
    )
    
    # DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"📊 Dataset: {len(dataset)} sequences")
    
    # Models
    print("\n🤖 Creating models...")
    generator = HighQualityGenerator(config)
    discriminator = MultiScaleDiscriminator()
    
    # Count parameters
    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"📊 Generator: {g_params:,} parameters")
    print(f"📊 Discriminator: {d_params:,} parameters")
    
    # Trainer
    trainer = HighQualityTrainer(generator, discriminator, config)
    
    # Train
    trainer.train(dataloader)
    
    print("\n✅ Training complete!")

if __name__ == "__main__":
    main()