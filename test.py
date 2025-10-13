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

# ==================== Configuration ====================
class Config:
    # Chemins
    dataset_path = "/kaggle/working/AnimaS/frames"  # Changez ceci vers votre dossier
    checkpoint_dir = "checkpoints"
    
    # Hyperparamètres
    image_size = 256  # Taille des images (256x256)
    sequence_length = 2  # Nombre d'images en entrée pour prédire la suivante
    latent_dim = 512  # Dimension de l'espace latent
    hidden_dim = 1024  # Dimension cachée
    n_heads = 8  # Nombre de têtes d'attention pour le Transformer
    n_layers = 4  # Nombre de couches du Transformer
    
    # Entraînement
    batch_size = 4  # Réduit pour 256x256
    num_epochs = 100
    learning_rate_g = 0.0002
    learning_rate_d = 0.0001
    beta1 = 0.5
    beta2 = 0.999
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Loss weights
    lambda_perceptual = 10.0
    lambda_l1 = 100.0
    lambda_gan = 1.0
    
    # Sauvegarde
    save_every = 5
    
config = Config()

# ==================== Perceptual Loss ====================
class PerceptualLoss(nn.Module):
    """Loss perceptuelle utilisant VGG16."""
    
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        
        # Charger VGG16 pré-entraîné
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features.to(device).eval()
        
        # Extraire les couches pour la loss perceptuelle
        self.layers = nn.ModuleList([
            nn.Sequential(*vgg[:4]),   # relu1_2
            nn.Sequential(*vgg[4:9]),  # relu2_2
            nn.Sequential(*vgg[9:16]), # relu3_3
            nn.Sequential(*vgg[16:23]) # relu4_3
        ])
        
        # Geler les paramètres
        for param in self.parameters():
            param.requires_grad = False
        
        # Poids pour chaque couche
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4]
        
    def forward(self, pred, target):
        """Calculer la loss perceptuelle entre prédiction et cible."""
        loss = 0
        
        # Dénormaliser de [-1, 1] à [0, 1] puis normaliser pour VGG
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        
        pred_vgg = (pred + 1) / 2  # [-1, 1] -> [0, 1]
        target_vgg = (target + 1) / 2
        
        pred_vgg = (pred_vgg - mean) / std
        target_vgg = (target_vgg - mean) / std
        
        x_pred = pred_vgg
        x_target = target_vgg
        
        for layer, weight in zip(self.layers, self.weights):
            x_pred = layer(x_pred)
            x_target = layer(x_target)
            loss += weight * nn.functional.l1_loss(x_pred, x_target)
        
        return loss

# ==================== Dataset avec Augmentation ====================
class FrameSequenceDataset(Dataset):
    """Dataset pour charger des séquences d'images avec augmentation."""
    
    def __init__(self, root_dir, sequence_length=2, transform=None, augment=False):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.augment = augment
        
        # Lister et trier toutes les images
        self.frames = sorted([f for f in os.listdir(root_dir) 
                             if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        if len(self.frames) < sequence_length + 1:
            raise ValueError(f"Pas assez d'images. Minimum requis: {sequence_length + 1}")
        
        print(f"Dataset initialisé avec {len(self.frames)} images")
        
        # Augmentation
        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        ])
        
    def __len__(self):
        return len(self.frames) - self.sequence_length
    
    def __getitem__(self, idx):
        input_sequence = []
        
        # Appliquer la même augmentation à toute la séquence
        if self.augment and random.random() > 0.5:
            apply_augment = True
            seed = random.randint(0, 2**32)
        else:
            apply_augment = False
        
        # Charger les images d'entrée
        for i in range(self.sequence_length):
            img_path = os.path.join(self.root_dir, self.frames[idx + i])
            image = Image.open(img_path).convert('RGB')
            
            if apply_augment:
                torch.manual_seed(seed)
                image = self.augmentation(image)
            
            if self.transform:
                image = self.transform(image)
            
            input_sequence.append(image)
        
        # Charger l'image cible
        target_path = os.path.join(self.root_dir, self.frames[idx + self.sequence_length])
        target_image = Image.open(target_path).convert('RGB')
        
        if apply_augment:
            torch.manual_seed(seed)
            target_image = self.augmentation(target_image)
        
        if self.transform:
            target_image = self.transform(target_image)
        
        input_tensor = torch.stack(input_sequence)
        
        return input_tensor, target_image

# ==================== Positional Encoding pour Transformer ====================
class PositionalEncoding(nn.Module):
    """Encodage positionnel pour le Transformer."""
    
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

# ==================== Modèle avec Transformer ====================
class ImageEncoder(nn.Module):
    """Encodeur CNN pour extraire les features des images 256x256."""
    
    def __init__(self, latent_dim=512):
        super(ImageEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            # Input: 3 x 256 x 256
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 64 x 128 x 128
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128 x 64 x 64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 256 x 32 x 32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 512 x 16 x 16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),  # 1024 x 8 x 8
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
        )
        
        # Corrected: 1024 * 8 * 8 = 65536 for 256x256 images
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 8 * 8, latent_dim),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x

class ImageDecoder(nn.Module):
    """Décodeur CNN pour reconstruire les images 256x256."""
    
    def __init__(self, latent_dim=512):
        super(ImageDecoder, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024 * 8 * 8),
            nn.LeakyReLU(0.2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # 512 x 16 x 16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 256 x 32 x 32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 128 x 64 x 64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64 x 128 x 128
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # 3 x 256 x 256
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 1024, 8, 8)
        x = self.decoder(x)
        return x

class FramePredictionGenerator(nn.Module):
    """Générateur avec Transformer pour prédire la prochaine frame."""
    
    def __init__(self, sequence_length=2, latent_dim=512, hidden_dim=1024, n_heads=8, n_layers=4):
        super(FramePredictionGenerator, self).__init__()
        
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        
        # Encodeur pour extraire les features
        self.encoder = ImageEncoder(latent_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(latent_dim)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)
        
        # Projection pour la prédiction
        self.prediction_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Décodeur pour reconstruire l'image
        self.decoder = ImageDecoder(latent_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        seq_length = x.size(1)
        
        # Encoder toutes les images de la séquence
        encoded_frames = []
        for i in range(seq_length):
            encoded = self.encoder(x[:, i])
            encoded_frames.append(encoded)
        
        # Stack pour créer une séquence
        encoded_sequence = torch.stack(encoded_frames, dim=1)
        
        # Ajouter l'encodage positionnel
        encoded_sequence = self.pos_encoder(encoded_sequence.transpose(0, 1)).transpose(0, 1)
        
        # Passer par le Transformer
        transformer_out = self.transformer_encoder(encoded_sequence)
        
        # Prendre la dernière sortie et prédire
        last_hidden = transformer_out[:, -1, :]
        predicted_latent = self.prediction_head(last_hidden)
        
        # Décoder pour obtenir l'image prédite
        predicted_frame = self.decoder(predicted_latent)
        
        return predicted_frame

# ==================== Discriminateur pour GAN ====================
class Discriminator(nn.Module):
    """Discriminateur PatchGAN pour améliorer le réalisme."""
    
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, normalization=True):
            """Bloc de base du discriminateur."""
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(input_channels * 2, 64, normalization=False),  # Concat input et target
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, kernel_size=4, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, img_A, img_B):
        # Concaténer les images conditionnelle et générée/réelle
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

# ==================== Fonctions d'entraînement GAN ====================
def train_epoch_gan(generator, discriminator, dataloader, 
                   criterion_gan, criterion_pixel, criterion_perceptual,
                   optimizer_g, optimizer_d, device, config):
    """Entraîner le modèle GAN pour une epoch."""
    generator.train()
    discriminator.train()
    
    total_g_loss = 0
    total_d_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training GAN")
    
    for batch_idx, (input_seq, target) in enumerate(progress_bar):
        input_seq = input_seq.to(device)
        target = target.to(device)
        
        # Prendre la dernière frame de la séquence comme condition
        condition = input_seq[:, -1]
        
        # ---------------------
        # Entraîner le Générateur
        # ---------------------
        optimizer_g.zero_grad()
        
        # Générer les images
        fake_frames = generator(input_seq)
        
        # Loss adversariale
        pred_fake = discriminator(condition, fake_frames)
        loss_gan = criterion_gan(pred_fake, torch.ones_like(pred_fake))
        
        # Loss pixel (L1)
        loss_pixel = criterion_pixel(fake_frames, target)
        
        # Loss perceptuelle
        loss_perceptual = criterion_perceptual(fake_frames, target)
        
        # Loss totale du générateur
        loss_g = (config.lambda_gan * loss_gan + 
                 config.lambda_l1 * loss_pixel + 
                 config.lambda_perceptual * loss_perceptual)
        
        loss_g.backward()
        optimizer_g.step()
        
        # ---------------------
        # Entraîner le Discriminateur
        # ---------------------
        optimizer_d.zero_grad()
        
        # Loss sur les vraies images
        pred_real = discriminator(condition, target)
        loss_real = criterion_gan(pred_real, torch.ones_like(pred_real))
        
        # Loss sur les fausses images
        pred_fake = discriminator(condition, fake_frames.detach())
        loss_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))
        
        # Loss totale du discriminateur
        loss_d = 0.5 * (loss_real + loss_fake)
        
        loss_d.backward()
        optimizer_d.step()
        
        total_g_loss += loss_g.item()
        total_d_loss += loss_d.item()
        
        # Mise à jour de la barre de progression
        progress_bar.set_postfix({
            'G_Loss': loss_g.item(),
            'D_Loss': loss_d.item(),
            'D_real': pred_real.mean().item(),
            'D_fake': pred_fake.mean().item()
        })
    
    return total_g_loss / len(dataloader), total_d_loss / len(dataloader)

def validate_gan(generator, dataloader, criterion_pixel, criterion_perceptual, device, config):
    """Valider le générateur."""
    generator.eval()
    total_loss = 0
    
    with torch.no_grad():
        for input_seq, target in tqdm(dataloader, desc="Validation"):
            input_seq = input_seq.to(device)
            target = target.to(device)
            
            predicted = generator(input_seq)
            
            loss_pixel = criterion_pixel(predicted, target)
            loss_perceptual = criterion_perceptual(predicted, target)
            
            loss = config.lambda_l1 * loss_pixel + config.lambda_perceptual * loss_perceptual
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

# ==================== Visualisation ====================
def visualize_prediction(generator, dataset, device, num_samples=3):
    """Visualiser les prédictions du modèle."""
    generator.eval()
    
    fig, axes = plt.subplots(num_samples, config.sequence_length + 2, 
                            figsize=(15, 5 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i in range(num_samples):
            idx = np.random.randint(0, len(dataset))
            input_seq, target = dataset[idx]
            
            input_batch = input_seq.unsqueeze(0).to(device)
            predicted = generator(input_batch)
            
            # Afficher les images d'entrée
            for j in range(config.sequence_length):
                img = input_seq[j].cpu().numpy().transpose(1, 2, 0)
                img = (img + 1) / 2
                img = np.clip(img, 0, 1)
                axes[i, j].imshow(img)
                axes[i, j].set_title(f"Input {j+1}")
                axes[i, j].axis('off')
            
            # Afficher la vraie image suivante
            target_img = target.cpu().numpy().transpose(1, 2, 0)
            target_img = (target_img + 1) / 2
            target_img = np.clip(target_img, 0, 1)
            axes[i, config.sequence_length].imshow(target_img)
            axes[i, config.sequence_length].set_title("Ground Truth")
            axes[i, config.sequence_length].axis('off')
            
            # Afficher la prédiction
            pred_img = predicted[0].cpu().numpy().transpose(1, 2, 0)
            pred_img = (pred_img + 1) / 2
            pred_img = np.clip(pred_img, 0, 1)
            axes[i, config.sequence_length + 1].imshow(pred_img)
            axes[i, config.sequence_length + 1].set_title("Prediction")
            axes[i, config.sequence_length + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
    plt.show()

# ==================== Fonction principale ====================
def main():
    # Créer le dossier de checkpoints
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Transformations pour les images
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Créer le dataset
    dataset = FrameSequenceDataset(
        root_dir=config.dataset_path,
        sequence_length=config.sequence_length,
        transform=transform,
        augment=True
    )
    
    # Diviser en train/validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Créer les dataloaders
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
    
    # Créer les modèles
    generator = FramePredictionGenerator(
        sequence_length=config.sequence_length,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
        n_heads=config.n_heads,
        n_layers=config.n_layers
    ).to(config.device)
    
    discriminator = Discriminator(input_channels=3).to(config.device)
    
    print(f"Générateur créé avec {sum(p.numel() for p in generator.parameters())} paramètres")
    print(f"Discriminateur créé avec {sum(p.numel() for p in discriminator.parameters())} paramètres")
    
    # Définir les losses
    criterion_gan = nn.BCELoss()
    criterion_pixel = nn.L1Loss()
    criterion_perceptual = PerceptualLoss(config.device)
    
    # Optimiseurs
    optimizer_g = optim.Adam(generator.parameters(), 
                            lr=config.learning_rate_g, 
                            betas=(config.beta1, config.beta2))
    optimizer_d = optim.Adam(discriminator.parameters(), 
                            lr=config.learning_rate_d, 
                            betas=(config.beta1, config.beta2))
    
    # Schedulers
    scheduler_g = optim.lr_scheduler.ReduceLROnPlateau(optimizer_g, mode='min', patience=5, factor=0.5)
    scheduler_d = optim.lr_scheduler.ReduceLROnPlateau(optimizer_d, mode='min', patience=5, factor=0.5)
    
    # Entraînement
    best_val_loss = float('inf')
    
    for epoch in range(1, config.num_epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{config.num_epochs}")
        print(f"{'='*50}")
        
        # Entraîner
        g_loss, d_loss = train_epoch_gan(
            generator, discriminator, train_loader,
            criterion_gan, criterion_pixel, criterion_perceptual,
            optimizer_g, optimizer_d, config.device, config
        )
        
        # Valider
        val_loss = validate_gan(
            generator, val_loader,
            criterion_pixel, criterion_perceptual,
            config.device, config
        )
        
        # Ajuster le learning rate
        scheduler_g.step(val_loss)
        scheduler_d.step(val_loss)
        
        print(f"G Loss: {g_loss:.4f} | D Loss: {d_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Sauvegarder le meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(config.checkpoint_dir, 'best_model.pth'))
            print(f"Meilleur modèle sauvegardé!")
        
        # Sauvegarder périodiquement
        if epoch % config.save_every == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'))
            
            # Visualiser les prédictions
            visualize_prediction(generator, val_dataset, config.device)
    
    print("\nEntraînement terminé!")

# ==================== Fonction de prédiction ====================
def predict_next_frame(model_path, frame_paths, output_path="predicted_frame.png"):
    """Prédire la prochaine frame à partir d'une séquence."""
    
    # Charger le modèle
    generator = FramePredictionGenerator(
        sequence_length=config.sequence_length,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
        n_heads=config.n_heads,
        n_layers=config.n_layers
    ).to(config.device)
    
    checkpoint = torch.load(model_path, map_location=config.device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    
    # Transformations
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Charger et transformer les images
    frames = []
    for path in frame_paths:
        img = Image.open(path).convert('RGB')
        img = transform(img)
        frames.append(img)
    
    # Stack et ajouter la dimension batch
    input_tensor = torch.stack(frames).unsqueeze(0).to(config.device)
    
    # Prédire
    with torch.no_grad():
        predicted = generator(input_tensor)
    
    # Convertir en image PIL
    pred_img = predicted[0].cpu().numpy().transpose(1, 2, 0)
    pred_img = (pred_img + 1) / 2
    pred_img = np.clip(pred_img, 0, 1)
    pred_img = (pred_img * 255).astype(np.uint8)
    
    result = Image.fromarray(pred_img)
    result.save(output_path)
    print(f"Prédiction sauvegardée: {output_path}")
    
    return result

if __name__ == "__main__":
    main()