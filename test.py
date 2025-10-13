import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import vgg16, VGG16_Weights
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import random
import math
import torch.nn.functional as F

# ==================== Configuration ====================
class Config:
    # Chemins
    dataset_path = "/kaggle/input/anima-s-dataset/test/"
    checkpoint_dir = "checkpoints"
    
    # Hyperparamètres
    image_size = 256
    sequence_length = 2
    latent_dim = 512
    hidden_dim = 1024
    n_heads = 8
    n_layers = 4
    
    # Entraînement - AJUSTEMENTS IMPORTANTS
    batch_size = 4
    num_epochs = 200
    learning_rate_g = 0.0001  # Réduit
    learning_rate_d = 0.0004  # Ratio 1:4 pour stabiliser
    beta1 = 0.5
    beta2 = 0.999
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Loss weights - AJUSTÉS
    lambda_perceptual = 1.0   # Réduit
    lambda_l1 = 10.0          # Augmenté pour forcer la reconstruction
    lambda_gan = 0.1          # Réduit pour éviter le mode collapse
    lambda_temporal = 5.0      # NOUVEAU: cohérence temporelle
    lambda_gradient = 2.0      # NOUVEAU: préservation des détails
    
    # Training strategy
    discriminator_steps = 1   # Steps du discriminateur par step du générateur
    generator_steps = 1
    warmup_epochs = 5         # Epochs sans GAN loss
    
    # Sauvegarde
    save_every = 5
    
config = Config()

# ==================== Temporal Consistency Loss ====================
class TemporalConsistencyLoss(nn.Module):
    """Loss pour maintenir la cohérence temporelle."""
    
    def __init__(self):
        super(TemporalConsistencyLoss, self).__init__()
        
    def forward(self, predicted, target, last_input):
        """
        Calcule la différence de mouvement entre:
        - Le mouvement prédit (last_input -> predicted)
        - Le mouvement réel (last_input -> target)
        """
        # Calculer les différences de frames (motion)
        predicted_motion = predicted - last_input
        target_motion = target - last_input
        
        # L1 loss sur le mouvement
        motion_loss = F.l1_loss(predicted_motion, target_motion)
        
        return motion_loss

# ==================== Gradient Loss ====================
class GradientLoss(nn.Module):
    """Loss pour préserver les détails et les contours."""
    
    def __init__(self):
        super(GradientLoss, self).__init__()
    
    def forward(self, pred, target):
        # Gradients horizontaux
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        
        # Gradients verticaux
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        # L1 loss sur les gradients
        dx_loss = F.l1_loss(pred_dx, target_dx)
        dy_loss = F.l1_loss(pred_dy, target_dy)
        
        return dx_loss + dy_loss

# ==================== Improved Perceptual Loss ====================
class ImprovedPerceptualLoss(nn.Module):
    """Loss perceptuelle améliorée avec plus de couches."""
    
    def __init__(self, device):
        super(ImprovedPerceptualLoss, self).__init__()
        
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features.to(device).eval()
        
        self.layers = nn.ModuleList([
            nn.Sequential(*vgg[:4]),   # relu1_2
            nn.Sequential(*vgg[4:9]),  # relu2_2
            nn.Sequential(*vgg[9:16]), # relu3_3
            nn.Sequential(*vgg[16:23]),# relu4_3
            nn.Sequential(*vgg[23:30]) # relu5_3
        ])
        
        for param in self.parameters():
            param.requires_grad = False
        
        # Poids ajustés pour chaque couche
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0/2]
        
    def forward(self, pred, target):
        loss = 0
        
        # Normalisation pour VGG
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        
        pred_vgg = (pred + 1) / 2
        target_vgg = (target + 1) / 2
        
        pred_vgg = (pred_vgg - mean) / std
        target_vgg = (target_vgg - mean) / std
        
        x_pred = pred_vgg
        x_target = target_vgg
        
        for layer, weight in zip(self.layers, self.weights):
            x_pred = layer(x_pred)
            x_target = layer(x_target)
            loss += weight * F.l1_loss(x_pred, x_target)
        
        return loss

# ==================== Residual Block ====================
class ResidualBlock(nn.Module):
    """Bloc résiduel pour améliorer le flux de gradient."""
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

# ==================== Improved Image Encoder ====================
class ImprovedImageEncoder(nn.Module):
    """Encodeur amélioré avec connexions résiduelles."""
    
    def __init__(self, latent_dim=512):
        super(ImprovedImageEncoder, self).__init__()
        
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        
        self.encoder = nn.ModuleList([
            # 256 -> 128
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                ResidualBlock(128)
            ),
            # 128 -> 64
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                ResidualBlock(256)
            ),
            # 64 -> 32
            nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2),
                ResidualBlock(512)
            ),
            # 32 -> 16
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2),
                ResidualBlock(512)
            ),
            # 16 -> 8
            nn.Sequential(
                nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.2)
            )
        ])
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 8 * 8, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1)  # Ajout de dropout
        )
        
    def forward(self, x):
        x = self.initial(x)
        for layer in self.encoder:
            x = layer(x)
        x = self.fc(x)
        return x

# ==================== Improved Image Decoder ====================
class ImprovedImageDecoder(nn.Module):
    """Décodeur amélioré avec connexions résiduelles."""
    
    def __init__(self, latent_dim=512):
        super(ImprovedImageDecoder, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024 * 8 * 8),
            nn.LayerNorm(1024 * 8 * 8),
            nn.LeakyReLU(0.2)
        )
        
        self.decoder = nn.ModuleList([
            # 8 -> 16
            nn.Sequential(
                nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2),
                ResidualBlock(512)
            ),
            # 16 -> 32
            nn.Sequential(
                nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2),
                ResidualBlock(512)
            ),
            # 32 -> 64
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                ResidualBlock(256)
            ),
            # 64 -> 128
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                ResidualBlock(128)
            ),
            # 128 -> 256
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2)
            )
        ])
        
        self.final = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 1024, 8, 8)
        for layer in self.decoder:
            x = layer(x)
        x = self.final(x)
        return x

# ==================== Improved Generator with Motion Modeling ====================
class ImprovedFramePredictionGenerator(nn.Module):
    """Générateur amélioré avec modélisation explicite du mouvement."""
    
    def __init__(self, sequence_length=2, latent_dim=512, hidden_dim=1024, n_heads=8, n_layers=4):
        super(ImprovedFramePredictionGenerator, self).__init__()
        
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        
        # Encodeur amélioré
        self.encoder = ImprovedImageEncoder(latent_dim)
        
        # Motion encoder - capture explicitement le mouvement
        self.motion_encoder = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(latent_dim)
        
        # Transformer avec plus d'attention
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,  # Plus de capacité
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)
        
        # Prediction head avec skip connection
        self.prediction_head = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),  # Concat avec motion
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        # Décodeur amélioré
        self.decoder = ImprovedImageDecoder(latent_dim)
        
        # Noise injection pour éviter le mode collapse
        self.noise_weight = 0.05
        
    def forward(self, x, inject_noise=True):
        batch_size = x.size(0)
        seq_length = x.size(1)
        
        # Encoder toutes les images
        encoded_frames = []
        for i in range(seq_length):
            encoded = self.encoder(x[:, i])
            
            # Injecter du bruit pendant l'entraînement
            if inject_noise and self.training:
                noise = torch.randn_like(encoded) * self.noise_weight
                encoded = encoded + noise
                
            encoded_frames.append(encoded)
        
        # Calculer le vecteur de mouvement
        if seq_length >= 2:
            motion_vector = self.motion_encoder(
                torch.cat([encoded_frames[-2], encoded_frames[-1]], dim=1)
            )
        else:
            motion_vector = torch.zeros_like(encoded_frames[0])
        
        # Stack pour créer une séquence
        encoded_sequence = torch.stack(encoded_frames, dim=1)
        
        # Ajouter l'encodage positionnel
        encoded_sequence = self.pos_encoder(encoded_sequence.transpose(0, 1)).transpose(0, 1)
        
        # Passer par le Transformer
        transformer_out = self.transformer_encoder(encoded_sequence)
        
        # Combiner la dernière sortie avec le vecteur de mouvement
        last_hidden = transformer_out[:, -1, :]
        combined = torch.cat([last_hidden, motion_vector], dim=1)
        
        # Prédire le prochain état latent
        predicted_latent = self.prediction_head(combined)
        
        # Ajouter une connexion résiduelle avec le dernier frame encodé
        predicted_latent = predicted_latent + encoded_frames[-1] * 0.5
        
        # Décoder
        predicted_frame = self.decoder(predicted_latent)
        
        return predicted_frame

# ==================== Improved Discriminator ====================
class ImprovedDiscriminator(nn.Module):
    """Discriminateur amélioré avec gradient penalty."""
    
    def __init__(self, input_channels=3):
        super(ImprovedDiscriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout2d(0.25))  # Ajout de dropout
            return layers
        
        # Input: concatenation of condition and target/generated
        self.model = nn.Sequential(
            *discriminator_block(input_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
            # Pas de sigmoid pour WGAN-GP
        )
    
    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

# ==================== Gradient Penalty for WGAN-GP ====================
def compute_gradient_penalty(discriminator, real_samples, fake_samples, condition, device):
    """Calcule le gradient penalty pour WGAN-GP."""
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    d_interpolates = discriminator(condition, interpolates)
    
    # Moyenne sur les dimensions spatiales pour obtenir une valeur par échantillon
    d_interpolates = d_interpolates.mean(dim=[2, 3])  # [batch_size, 1]
    
    fake = torch.ones_like(d_interpolates, device=device, requires_grad=False)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty

# ==================== Training Functions ====================
def train_epoch_improved(generator, discriminator, dataloader, 
                         criterion_pixel, criterion_perceptual, criterion_temporal,
                         criterion_gradient, optimizer_g, optimizer_d, 
                         device, config, epoch):
    """Fonction d'entraînement améliorée."""
    generator.train()
    discriminator.train()
    
    total_g_loss = 0
    total_d_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch}")
    
    for batch_idx, (input_seq, target) in enumerate(progress_bar):
        input_seq = input_seq.to(device)
        target = target.to(device)
        
        # Prendre la dernière frame comme condition
        condition = input_seq[:, -1]
        
        # ---------------------
        # Entraîner le Discriminateur
        # ---------------------
        for _ in range(config.discriminator_steps):
            optimizer_d.zero_grad()
            
            # Générer les fausses images
            with torch.no_grad():
                fake_frames = generator(input_seq, inject_noise=True)
            
            # Scores pour les vraies et fausses images
            pred_real = discriminator(condition, target)
            pred_fake = discriminator(condition, fake_frames)
            
            # WGAN loss
            d_loss_real = -torch.mean(pred_real)
            d_loss_fake = torch.mean(pred_fake)
            
            # Gradient penalty
            gp = compute_gradient_penalty(discriminator, target, fake_frames, condition, device)
            
            # Loss totale du discriminateur
            d_loss = d_loss_real + d_loss_fake + 10 * gp  # Lambda GP = 10
            
            d_loss.backward()
            optimizer_d.step()
        
        # ---------------------
        # Entraîner le Générateur
        # ---------------------
        for _ in range(config.generator_steps):
            optimizer_g.zero_grad()
            
            # Générer les images
            fake_frames = generator(input_seq, inject_noise=True)
            
            # Loss adversariale (WGAN)
            pred_fake = discriminator(condition, fake_frames)
            loss_gan = -torch.mean(pred_fake) if epoch > config.warmup_epochs else 0
            
            # Loss pixel (L1)
            loss_pixel = criterion_pixel(fake_frames, target)
            
            # Loss perceptuelle
            loss_perceptual = criterion_perceptual(fake_frames, target)
            
            # Loss temporelle
            loss_temporal = criterion_temporal(fake_frames, target, condition)
            
            # Loss gradient
            loss_gradient = criterion_gradient(fake_frames, target)
            
            # Loss totale du générateur
            if epoch <= config.warmup_epochs:
                # Pendant le warmup, pas de GAN loss
                loss_g = (config.lambda_l1 * loss_pixel + 
                         config.lambda_perceptual * loss_perceptual +
                         config.lambda_temporal * loss_temporal +
                         config.lambda_gradient * loss_gradient)
            else:
                loss_g = (config.lambda_gan * loss_gan + 
                         config.lambda_l1 * loss_pixel + 
                         config.lambda_perceptual * loss_perceptual +
                         config.lambda_temporal * loss_temporal +
                         config.lambda_gradient * loss_gradient)
            
            loss_g.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            
            optimizer_g.step()
        
        total_g_loss += loss_g.item()
        total_d_loss += d_loss.item() if 'd_loss' in locals() else 0
        
        # Mise à jour de la barre de progression
        progress_bar.set_postfix({
            'G_Loss': f"{loss_g.item():.4f}",
            'D_Loss': f"{d_loss.item():.4f}" if 'd_loss' in locals() else "N/A",
            'L1': f"{loss_pixel.item():.4f}",
            'Perc': f"{loss_perceptual.item():.4f}"
        })
    
    return total_g_loss / len(dataloader), total_d_loss / len(dataloader)

# ==================== Positional Encoding ====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# ==================== Main Training Loop ====================
def main():
    # Créer le dossier de checkpoints
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Transformations
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Dataset
    from glob import glob
    print(f"Recherche des images dans: {config.dataset_path}")
    image_files = glob(os.path.join(config.dataset_path, "*.png")) + \
                  glob(os.path.join(config.dataset_path, "*.jpg"))
    print(f"Nombre d'images trouvées: {len(image_files)}")
    
    dataset = FrameSequenceDataset(
        root_dir=config.dataset_path,
        sequence_length=config.sequence_length,
        transform=transform,
        augment=True
    )
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Modèles
    generator = ImprovedFramePredictionGenerator(
        sequence_length=config.sequence_length,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
        n_heads=config.n_heads,
        n_layers=config.n_layers
    ).to(config.device)
    
    discriminator = ImprovedDiscriminator(input_channels=3).to(config.device)
    
    print(f"Générateur: {sum(p.numel() for p in generator.parameters())} paramètres")
    print(f"Discriminateur: {sum(p.numel() for p in discriminator.parameters())} paramètres")
    
    # Losses
    criterion_pixel = nn.L1Loss()
    criterion_perceptual = ImprovedPerceptualLoss(config.device)
    criterion_temporal = TemporalConsistencyLoss()
    criterion_gradient = GradientLoss()
    
    # Optimizers
    optimizer_g = optim.AdamW(generator.parameters(), 
                              lr=config.learning_rate_g, 
                              betas=(config.beta1, config.beta2),
                              weight_decay=0.01)
    
    optimizer_d = optim.AdamW(discriminator.parameters(), 
                              lr=config.learning_rate_d, 
                              betas=(config.beta1, config.beta2),
                              weight_decay=0.01)
    
    # Learning rate schedulers
    scheduler_g = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_g, T_0=10, T_mult=2)
    scheduler_d = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_d, T_0=10, T_mult=2)
    
    # Training
    best_val_loss = float('inf')
    
    for epoch in range(1, config.num_epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{config.num_epochs}")
        print(f"LR - G: {optimizer_g.param_groups[0]['lr']:.6f}, D: {optimizer_d.param_groups[0]['lr']:.6f}")
        print(f"{'='*50}")
        
        # Train
        g_loss, d_loss = train_epoch_improved(
            generator, discriminator, train_loader,
            criterion_pixel, criterion_perceptual, criterion_temporal,
            criterion_gradient, optimizer_g, optimizer_d,
            config.device, config, epoch
        )
        
        # Validation
        generator.eval()
        val_loss = 0
        with torch.no_grad():
            for input_seq, target in tqdm(val_loader, desc="Validation"):
                input_seq = input_seq.to(config.device)
                target = target.to(config.device)
                
                predicted = generator(input_seq, inject_noise=False)
                
                loss = criterion_pixel(predicted, target) + \
                       criterion_perceptual(predicted, target) * 0.1
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Update learning rates
        scheduler_g.step()
        scheduler_d.step()
        
        print(f"G Loss: {g_loss:.4f} | D Loss: {d_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(config.checkpoint_dir, 'best_model.pth'))
            print(f"✓ Meilleur modèle sauvegardé!")
        
        # Save periodically
        if epoch % config.save_every == 0:
            # Visualisation
            visualize_prediction_improved(generator, val_dataset, config.device, epoch)
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
            }, os.path.join(config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    print("\n✓ Entraînement terminé!")

# ==================== Improved Visualization ====================
def visualize_prediction_improved(generator, dataset, device, epoch):
    """Visualisation améliorée avec plus de détails."""
    generator.eval()
    
    num_samples = min(3, len(dataset))
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i in range(num_samples):
            idx = np.random.randint(0, len(dataset))
            input_seq, target = dataset[idx]
            
            input_batch = input_seq.unsqueeze(0).to(device)
            predicted = generator(input_batch, inject_noise=False)
            
            # Input frames
            for j in range(config.sequence_length):
                img = input_seq[j].cpu().numpy().transpose(1, 2, 0)
                img = (img + 1) / 2
                img = np.clip(img, 0, 1)
                axes[i, j].imshow(img)
                axes[i, j].set_title(f"Input {j+1}")
                axes[i, j].axis('off')
            
            # Ground truth
            target_img = target.cpu().numpy().transpose(1, 2, 0)
            target_img = (target_img + 1) / 2
            target_img = np.clip(target_img, 0, 1)
            axes[i, 2].imshow(target_img)
            axes[i, 2].set_title("Ground Truth")
            axes[i, 2].axis('off')
            
            # Prediction
            pred_img = predicted[0].cpu().numpy().transpose(1, 2, 0)
            pred_img = (pred_img + 1) / 2
            pred_img = np.clip(pred_img, 0, 1)
            axes[i, 3].imshow(pred_img)
            axes[i, 3].set_title("Prediction")
            axes[i, 3].axis('off')
            
            # Difference map
            diff = np.abs(target_img - pred_img).mean(axis=2)
            im = axes[i, 4].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
            axes[i, 4].set_title("Difference")
            axes[i, 4].axis('off')
            plt.colorbar(im, ax=axes[i, 4], fraction=0.046)
    
    plt.suptitle(f'Epoch {epoch} - Frame Predictions', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'predictions_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
    plt.close()

# ==================== Dataset Class ====================
class FrameSequenceDataset(Dataset):
    def __init__(self, root_dir, sequence_length=2, transform=None, augment=False):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.augment = augment
        
        self.frames = sorted([f for f in os.listdir(root_dir) 
                             if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        if len(self.frames) < sequence_length + 1:
            raise ValueError(f"Pas assez d'images. Minimum: {sequence_length + 1}")
        
        print(f"Dataset: {len(self.frames)} images")
        
    def __len__(self):
        return len(self.frames) - self.sequence_length
    
    def __getitem__(self, idx):
        input_sequence = []
        
        for i in range(self.sequence_length):
            img_path = os.path.join(self.root_dir, self.frames[idx + i])
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            input_sequence.append(image)
        
        target_path = os.path.join(self.root_dir, self.frames[idx + self.sequence_length])
        target_image = Image.open(target_path).convert('RGB')
        
        if self.transform:
            target_image = self.transform(target_image)
        
        input_tensor = torch.stack(input_sequence)
        
        return input_tensor, target_image

if __name__ == "__main__":
    main()