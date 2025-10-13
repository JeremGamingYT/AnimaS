import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from typing import List, Tuple
import traceback

# ==================== 1. DATASET POUR VOS IMAGES ====================
class VideoSequenceDataset(Dataset):
    """
    Dataset pour charger VOS séquences d'images
    
    Structures supportées:
    
    Option 1 - Plusieurs séquences dans des sous-dossiers:
        data/
            seq_001/
                frame_0001.png
                frame_0002.png
                frame_0003.png
            seq_002/
                frame_0001.png
                frame_0002.png
    
    Option 2 - Une seule séquence (toutes les images dans un dossier):
        data/
            frame_0001.png
            frame_0002.png
            frame_0003.png
            
    Option 3 - Vidéo continue (les noms peuvent être quelconques):
        data/
            img001.jpg
            img002.jpg
            img003.jpg
    """
    def __init__(self, data_dir, context_length=2, img_size=64, stride=1, extensions=None):
        """
        Args:
            data_dir: Chemin vers vos images
            context_length: Nombre d'images en entrée pour prédire la suivante
            img_size: Taille de redimensionnement des images
            stride: Pas entre les échantillons (1 = tous, 2 = un sur deux, etc.)
            extensions: Liste d'extensions acceptées (ex: ['.png', '.jpg', '.jpeg'])
        """
        self.data_dir = Path(data_dir)
        self.context_length = context_length
        self.img_size = img_size
        self.stride = stride
        
        if extensions is None:
            self.extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.PNG', '.JPG', '.JPEG']
        else:
            self.extensions = extensions
        
        self.samples = []
        self._load_dataset()
        
        if len(self.samples) == 0:
            raise ValueError(f"❌ Aucune séquence trouvée dans {data_dir}!\n"
                           f"   Vérifiez que vos images sont bien organisées.")
    
    def _load_dataset(self):
        """Détecte automatiquement la structure et charge les échantillons"""
        
        # Vérifier si c'est une structure avec sous-dossiers
        subdirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        if subdirs:
            # Option 1: Plusieurs séquences dans des sous-dossiers
            print(f"📁 Détection: Structure avec sous-dossiers ({len(subdirs)} séquences)")
            for seq_dir in sorted(subdirs):
                self._process_sequence(seq_dir)
        else:
            # Option 2 ou 3: Une seule séquence
            print(f"📁 Détection: Structure plate (une seule séquence)")
            self._process_sequence(self.data_dir)
        
        print(f"✅ {len(self.samples)} échantillons chargés")
    
    def _process_sequence(self, seq_dir):
        """Traite une séquence d'images"""
        # Trouver toutes les images
        frames = []
        for ext in self.extensions:
            frames.extend(list(seq_dir.glob(f"*{ext}")))
        
        # Trier par nom
        frames = sorted(frames)
        
        if len(frames) < self.context_length + 1:
            print(f"⚠️  Séquence {seq_dir.name} ignorée: seulement {len(frames)} images "
                  f"(minimum requis: {self.context_length + 1})")
            return
        
        # Créer des échantillons
        for i in range(0, len(frames) - self.context_length, self.stride):
            if i + self.context_length >= len(frames):
                break
            
            self.samples.append({
                'input_frames': frames[i:i+self.context_length],
                'target_frame': frames[i+self.context_length],
                'sequence_name': seq_dir.name if seq_dir != self.data_dir else 'main'
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Charger les images d'entrée
        input_imgs = []
        for frame_path in sample['input_frames']:
            img = Image.open(frame_path).convert('RGB')
            img = img.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
            img = np.array(img).astype(np.float32) / 255.0
            input_imgs.append(img)
        
        # Charger l'image cible
        target_img = Image.open(sample['target_frame']).convert('RGB')
        target_img = target_img.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        target_img = np.array(target_img).astype(np.float32) / 255.0
        
        # Convertir en tenseurs (B, C, H, W)
        input_imgs = torch.FloatTensor(np.array(input_imgs)).permute(0, 3, 1, 2)
        target_img = torch.FloatTensor(target_img).permute(2, 0, 1)
        
        return input_imgs, target_img
    
    def get_sample_info(self, idx):
        """Retourne les informations sur un échantillon"""
        return self.samples[idx]

# ==================== 2. FONCTIONS UTILITAIRES POUR VÉRIFIER LE DATASET ====================
def analyze_dataset(data_dir):
    """Analyse votre dataset et affiche des informations"""
    data_dir = Path(data_dir)
    
    print("\n" + "="*60)
    print("📊 ANALYSE DU DATASET")
    print("="*60)
    
    if not data_dir.exists():
        print(f"❌ Le dossier {data_dir} n'existe pas!")
        return False
    
    # Compter les fichiers images
    extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.PNG', '.JPG', '.JPEG']
    all_images = []
    for ext in extensions:
        all_images.extend(list(data_dir.rglob(f"*{ext}")))
    
    print(f"\n📁 Dossier: {data_dir}")
    print(f"📸 Nombre total d'images: {len(all_images)}")
    
    # Vérifier la structure
    subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    if subdirs:
        print(f"📂 Structure: Sous-dossiers détectés ({len(subdirs)} séquences)")
        print(f"\n   Liste des séquences:")
        for i, subdir in enumerate(sorted(subdirs)[:10]):
            imgs = []
            for ext in extensions:
                imgs.extend(list(subdir.glob(f"*{ext}")))
            print(f"   - {subdir.name}: {len(imgs)} images")
        if len(subdirs) > 10:
            print(f"   ... et {len(subdirs) - 10} autres séquences")
    else:
        print(f"📂 Structure: Séquence unique (toutes les images dans le même dossier)")
    
    # Analyser quelques images
    if all_images:
        print(f"\n🖼️  Analyse d'images d'exemple:")
        sample_img = Image.open(all_images[0])
        print(f"   - Résolution: {sample_img.size}")
        print(f"   - Mode: {sample_img.mode}")
        print(f"   - Format: {sample_img.format}")
    
    print("\n" + "="*60)
    return True

def visualize_dataset_samples(dataset, num_samples=3, save_path="dataset_preview.png"):
    """Visualise quelques échantillons du dataset"""
    fig, axes = plt.subplots(num_samples, dataset.context_length + 1, 
                             figsize=(3 * (dataset.context_length + 1), 3 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        idx = np.random.randint(0, len(dataset))
        input_frames, target_frame = dataset[idx]
        info = dataset.get_sample_info(idx)
        
        # Afficher les frames d'entrée
        for j in range(dataset.context_length):
            axes[i, j].imshow(input_frames[j].permute(1, 2, 0).numpy())
            axes[i, j].set_title(f"Input {j+1}")
            axes[i, j].axis('off')
        
        # Afficher la frame cible
        axes[i, dataset.context_length].imshow(target_frame.permute(1, 2, 0).numpy())
        axes[i, dataset.context_length].set_title("Target")
        axes[i, dataset.context_length].axis('off')
        
        # Ajouter le nom de la séquence
        fig.text(0.02, 1 - (i + 0.5) / num_samples, 
                f"Seq: {info['sequence_name']}", 
                fontsize=10, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Aperçu sauvegardé: {save_path}")
    plt.close()

# ==================== 3. MODÈLE DE TOKENISATION ====================
class ImageTokenizer(nn.Module):
    """Encode les images en représentations latentes - VERSION ADAPTATIVE"""
    def __init__(self, img_size=64, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        
        # Calculer le nombre de downsampling nécessaires
        # Pour arriver à une taille raisonnable (8x8 ou 16x16)
        if img_size >= 256:
            # Pour 256: 256 -> 128 -> 64 -> 32 -> 16
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, 1),      # 256 -> 128
                nn.ReLU(),
                nn.Conv2d(64, 128, 4, 2, 1),    # 128 -> 64
                nn.ReLU(),
                nn.Conv2d(128, 256, 4, 2, 1),   # 64 -> 32
                nn.ReLU(),
                nn.Conv2d(256, 256, 4, 2, 1),   # 32 -> 16
                nn.ReLU(),
                nn.Conv2d(256, embed_dim, 3, 1, 1),  # 16 -> 16
            )
            
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, 256, 3, 1, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 256, 4, 2, 1),   # 16 -> 32
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, 4, 2, 1),   # 32 -> 64
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),    # 64 -> 128
                nn.ReLU(),
                nn.ConvTranspose2d(64, 3, 4, 2, 1),      # 128 -> 256
                nn.Sigmoid()
            )
            self.latent_size = 16
            
        elif img_size >= 128:
            # Pour 128: 128 -> 64 -> 32 -> 16
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, 1),      # 128 -> 64
                nn.ReLU(),
                nn.Conv2d(64, 128, 4, 2, 1),    # 64 -> 32
                nn.ReLU(),
                nn.Conv2d(128, 256, 4, 2, 1),   # 32 -> 16
                nn.ReLU(),
                nn.Conv2d(256, embed_dim, 3, 1, 1),  # 16 -> 16
            )
            
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, 256, 3, 1, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, 4, 2, 1),   # 16 -> 32
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),    # 32 -> 64
                nn.ReLU(),
                nn.ConvTranspose2d(64, 3, 4, 2, 1),      # 64 -> 128
                nn.Sigmoid()
            )
            self.latent_size = 16
            
        else:  # 64 ou moins
            # Pour 64: 64 -> 32 -> 16 -> 8
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, 1),      # 64 -> 32
                nn.ReLU(),
                nn.Conv2d(64, 128, 4, 2, 1),    # 32 -> 16
                nn.ReLU(),
                nn.Conv2d(128, 256, 4, 2, 1),   # 16 -> 8
                nn.ReLU(),
                nn.Conv2d(256, embed_dim, 3, 1, 1),  # 8 -> 8
            )
            
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, 256, 3, 1, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, 4, 2, 1),   # 8 -> 16
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),    # 16 -> 32
                nn.ReLU(),
                nn.ConvTranspose2d(64, 3, 4, 2, 1),      # 32 -> 64
                nn.Sigmoid()
            )
            self.latent_size = 8
        
        print(f"   🔍 Tokenizer: {img_size}x{img_size} -> {self.latent_size}x{self.latent_size} latent")
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

# ==================== 4. MODÈLE TRANSFORMER ====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)

class VideoGPT(nn.Module):
    """Modèle type GPT pour prédire la prochaine image - VERSION CORRIGÉE"""
    def __init__(self, embed_dim=256, num_heads=8, num_layers=6, img_size=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = img_size
        
        # Tokenizer d'images
        self.tokenizer = ImageTokenizer(img_size=img_size, embed_dim=embed_dim)
        
        # Calculer la dimension après flattening
        self.latent_size = self.tokenizer.latent_size
        self.latent_dim = embed_dim * self.latent_size * self.latent_size
        
        print(f"   🔍 Latent dimension: {self.latent_dim} ({embed_dim} x {self.latent_size} x {self.latent_size})")
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.latent_dim)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim, 
            nhead=num_heads,
            dim_feedforward=self.latent_dim * 2,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Projection pour décoder
        self.projection = nn.Linear(self.latent_dim, self.latent_dim)
    
    def forward(self, input_frames):
        """
        input_frames: (batch, num_frames, 3, H, W)
        """
        batch_size, num_frames, C, H, W = input_frames.shape
        
        # Encoder chaque frame
        encoded_frames = []
        for i in range(num_frames):
            _, z = self.tokenizer(input_frames[:, i])  # (B, embed_dim, latent_size, latent_size)
            z = z.flatten(1)  # (B, latent_dim)
            encoded_frames.append(z)
        
        # Stack les frames encodés
        encoded_seq = torch.stack(encoded_frames, dim=1)  # (B, num_frames, latent_dim)
        
        # Positional encoding
        encoded_seq = self.pos_encoder(encoded_seq)
        
        # Transformer
        transformed = self.transformer(encoded_seq)  # (B, num_frames, latent_dim)
        
        # Prendre la dernière sortie pour prédire la prochaine frame
        next_frame_encoding = self.projection(transformed[:, -1])  # (B, latent_dim)
        
        # Reshape et décoder
        next_frame_encoding = next_frame_encoding.view(
            batch_size, self.embed_dim, self.latent_size, self.latent_size
        )
        predicted_frame = self.tokenizer.decode(next_frame_encoding)
        
        return predicted_frame

# ==================== 5. ENTRAÎNEMENT AVEC VISUALISATION ====================
def train_model(model, train_loader, val_loader=None, num_epochs=50, lr=1e-4, 
                device='cuda', save_dir="checkpoints", visualize_every=5):
    """Entraîne le modèle avec sauvegarde d'images à chaque epoch"""
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "predictions"), exist_ok=True)
    
    print(f"\n🔍 DEBUG - Préparation de l'entraînement...")
    print(f"   Device: {device}")
    print(f"   Nombre d'epochs: {num_epochs}")
    print(f"   Learning rate: {lr}")
    
    try:
        model = model.to(device)
        print(f"   ✅ Modèle transféré sur {device}")
    except Exception as e:
        print(f"   ❌ Erreur lors du transfert sur {device}: {e}")
        raise
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    print(f"\n🚀 Démarrage de l'entraînement...\n")
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"📅 EPOCH {epoch+1}/{num_epochs}")
        print(f"{'='*60}")
        
        # ===== ENTRAÎNEMENT =====
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        try:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [TRAIN]")
            for batch_idx, (input_frames, target_frame) in enumerate(pbar):
                try:
                    input_frames = input_frames.to(device)
                    target_frame = target_frame.to(device)
                    
                    # Forward pass
                    predicted_frame = model(input_frames)
                    loss = criterion(predicted_frame, target_frame)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    batch_count += 1
                    pbar.set_postfix({'loss': f'{loss.item():.6f}'})
                    
                except Exception as e:
                    print(f"\n❌ Erreur au batch {batch_idx}: {e}")
                    print(f"   Input shape: {input_frames.shape}")
                    print(f"   Target shape: {target_frame.shape}")
                    traceback.print_exc()
                    raise
            
            avg_train_loss = epoch_loss / batch_count if batch_count > 0 else 0
            history['train_loss'].append(avg_train_loss)
            print(f"✅ Train Loss: {avg_train_loss:.6f}")
            
        except Exception as e:
            print(f"\n❌ Erreur pendant l'entraînement à l'epoch {epoch+1}: {e}")
            traceback.print_exc()
            raise
        
        # ===== VALIDATION =====
        if val_loader is not None:
            model.eval()
            val_loss = 0
            val_batch_count = 0
            
            with torch.no_grad():
                pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [VAL]")
                for input_frames, target_frame in pbar_val:
                    input_frames = input_frames.to(device)
                    target_frame = target_frame.to(device)
                    predicted_frame = model(input_frames)
                    val_loss += criterion(predicted_frame, target_frame).item()
                    val_batch_count += 1
                    pbar_val.set_postfix({'val_loss': f'{val_loss/val_batch_count:.6f}'})
            
            avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else 0
            history['val_loss'].append(avg_val_loss)
            scheduler.step(avg_val_loss)
            
            print(f"✅ Val Loss: {avg_val_loss:.6f}")
            
            # Sauvegarder le meilleur modèle
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_val_loss,
                }, os.path.join(save_dir, 'best_model.pth'))
                print(f"   💾 Meilleur modèle sauvegardé (val_loss: {avg_val_loss:.6f})")
        
        # ===== VISUALISATION =====
        if (epoch + 1) % visualize_every == 0 or epoch == 0:
            print(f"🖼️  Génération de visualisations...")
            try:
                model.eval()
                with torch.no_grad():
                    # Prendre un échantillon aléatoire
                    sample_input, sample_target = next(iter(val_loader if val_loader else train_loader))
                    sample_input = sample_input[:4].to(device)  # Prendre 4 exemples
                    sample_target = sample_target[:4].to(device)
                    sample_pred = model(sample_input)
                    
                    # Créer la visualisation
                    fig, axes = plt.subplots(4, sample_input.shape[1] + 2, figsize=(3*(sample_input.shape[1]+2), 12))
                    
                    for i in range(4):
                        # Frames d'entrée
                        for j in range(sample_input.shape[1]):
                            img = sample_input[i, j].cpu().permute(1, 2, 0).numpy()
                            axes[i, j].imshow(img)
                            axes[i, j].set_title(f"Input {j+1}")
                            axes[i, j].axis('off')
                        
                        # Target
                        img_target = sample_target[i].cpu().permute(1, 2, 0).numpy()
                        axes[i, sample_input.shape[1]].imshow(img_target)
                        axes[i, sample_input.shape[1]].set_title("Target")
                        axes[i, sample_input.shape[1]].axis('off')
                        axes[i, sample_input.shape[1]].set_facecolor('#e8f4ea')
                        
                        # Prédiction
                        img_pred = sample_pred[i].cpu().permute(1, 2, 0).numpy()
                        axes[i, sample_input.shape[1] + 1].imshow(img_pred)
                        axes[i, sample_input.shape[1] + 1].set_title("Predicted")
                        axes[i, sample_input.shape[1] + 1].axis('off')
                        axes[i, sample_input.shape[1] + 1].set_facecolor('#fff4e6')
                    
                    plt.suptitle(f'Epoch {epoch+1} - Train Loss: {avg_train_loss:.6f}', fontsize=16)
                    plt.tight_layout()
                    
                    save_path = os.path.join(save_dir, "predictions", f"epoch_{epoch+1:03d}.png")
                    plt.savefig(save_path, dpi=100, bbox_inches='tight')
                    plt.close()
                    print(f"   ✅ Visualisation sauvegardée: {save_path}")
                    
            except Exception as e:
                print(f"   ⚠️  Erreur lors de la visualisation: {e}")
        
        # Sauvegarder checkpoint régulier
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
            }, checkpoint_path)
            print(f"   💾 Checkpoint sauvegardé: {checkpoint_path}")
    
    print(f"\n{'='*60}")
    print("✅ ENTRAÎNEMENT TERMINÉ!")
    print(f"{'='*60}\n")
    
    return history

# ==================== 6. PRÉDICTION ====================
def predict_next_frame(model, input_frames, device='cuda'):
    """Prédit la prochaine image"""
    model.eval()
    with torch.no_grad():
        if len(input_frames.shape) == 4:  # Si pas de batch dimension
            input_frames = input_frames.unsqueeze(0)
        input_frames = input_frames.to(device)
        predicted_frame = model(input_frames)
    return predicted_frame.cpu().squeeze(0)

def predict_sequence(model, initial_frames, num_predictions=5, device='cuda'):
    """Prédit plusieurs images dans le futur"""
    predictions = []
    current_frames = initial_frames.clone()
    
    for i in range(num_predictions):
        next_frame = predict_next_frame(model, current_frames, device)
        predictions.append(next_frame)
        
        # Shift: retirer la première frame et ajouter la prédiction
        current_frames = torch.cat([current_frames[1:], next_frame.unsqueeze(0)], dim=0)
    
    return predictions

# ==================== 7. VISUALISATION ====================
def visualize_prediction(input_frames, target_frame, predicted_frame, save_path="prediction.png"):
    """Visualise une prédiction"""
    num_inputs = input_frames.shape[0]
    
    fig, axes = plt.subplots(1, num_inputs + 2, figsize=(3 * (num_inputs + 2), 3))
    
    # Images d'entrée
    for i in range(num_inputs):
        axes[i].imshow(input_frames[i].permute(1, 2, 0).numpy())
        axes[i].set_title(f"Input Frame {i+1}")
        axes[i].axis('off')
    
    # Image cible
    axes[num_inputs].imshow(target_frame.permute(1, 2, 0).numpy())
    axes[num_inputs].set_title("Target")
    axes[num_inputs].axis('off')
    
    # Image prédite
    axes[num_inputs + 1].imshow(predicted_frame.permute(1, 2, 0).numpy())
    axes[num_inputs + 1].set_title("Predicted")
    axes[num_inputs + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_sequence_prediction(initial_frames, predictions, save_path="sequence_prediction.png"):
    """Visualise une séquence de prédictions"""
    num_initial = len(initial_frames)
    num_pred = len(predictions)
    total = num_initial + num_pred
    
    fig, axes = plt.subplots(1, total, figsize=(3 * total, 3))
    
    # Frames initiales
    for i in range(num_initial):
        axes[i].imshow(initial_frames[i].permute(1, 2, 0).numpy())
        axes[i].set_title(f"Input {i+1}")
        axes[i].axis('off')
        axes[i].set_facecolor('#e8f4ea')
    
    # Prédictions
    for i, pred in enumerate(predictions):
        axes[num_initial + i].imshow(pred.permute(1, 2, 0).numpy())
        axes[num_initial + i].set_title(f"Pred {i+1}")
        axes[num_initial + i].axis('off')
        axes[num_initial + i].set_facecolor('#fff4e6')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Séquence sauvegardée: {save_path}")
    plt.close()

# ==================== 8. FONCTION PRINCIPALE ====================
def main():
    """Fonction principale"""
    
    # ========== CONFIGURATION ==========
    # 🔧 MODIFIEZ CES PARAMÈTRES SELON VOS BESOINS
    CONFIG = {
        'data_dir': '/kaggle/input/anima-s-dataset/test/',  # 📁 CHEMIN VERS VOS IMAGES
        'img_size': 256,             # Taille des images (64, 128, 256...)
        'context_length': 2,         # Nombre d'images en entrée
        'batch_size': 8,             # Taille des batchs (réduit pour img_size=256)
        'num_epochs': 50,            # Nombre d'époques
        'learning_rate': 1e-4,       # Taux d'apprentissage
        'val_split': 0.2,            # Proportion pour validation (0.2 = 20%)
        'stride': 1,                 # Pas entre échantillons (1 = tous)
        'embed_dim': 256,            # Dimension des embeddings
        'num_heads': 8,              # Nombre de têtes d'attention
        'num_layers': 4,             # Nombre de couches Transformer
        'visualize_every': 5,        # Sauvegarder des images tous les N epochs
    }
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"🚀 ENTRAÎNEMENT VIDEO-GPT")
    print(f"{'='*60}")
    print(f"💻 Device: {DEVICE}")
    print(f"📁 Dataset: {CONFIG['data_dir']}")
    print(f"🖼️  Taille images: {CONFIG['img_size']}x{CONFIG['img_size']}")
    print(f"📊 Context: {CONFIG['context_length']} frames")
    print(f"{'='*60}\n")
    
    try:
        # ========== 1. ANALYSER LE DATASET ==========
        print("📊 ÉTAPE 1/6: Analyse du dataset")
        print("-" * 60)
        if not analyze_dataset(CONFIG['data_dir']):
            print("\n❌ Veuillez corriger les problèmes ci-dessus avant de continuer.")
            return
        
        # ========== 2. CHARGER LE DATASET ==========
        print("\n📦 ÉTAPE 2/6: Chargement du dataset")
        print("-" * 60)
        
        full_dataset = VideoSequenceDataset(
            CONFIG['data_dir'],
            context_length=CONFIG['context_length'],
            img_size=CONFIG['img_size'],
            stride=CONFIG['stride']
        )
        
        # Split train/val
        val_size = int(len(full_dataset) * CONFIG['val_split'])
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=CONFIG['batch_size'], 
            shuffle=True, 
            num_workers=0,
            pin_memory=True if DEVICE == 'cuda' else False
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=CONFIG['batch_size'], 
            shuffle=False, 
            num_workers=0,
            pin_memory=True if DEVICE == 'cuda' else False
        )
        
        print(f"✅ Dataset chargé:")
        print(f"   - Total: {len(full_dataset)} échantillons")
        print(f"   - Train: {train_size} échantillons")
        print(f"   - Val: {val_size} échantillons")
        print(f"   - Batch size: {CONFIG['batch_size']}")
        print(f"   - Train batches: {len(train_loader)}")
        print(f"   - Val batches: {len(val_loader)}")
        
        # ========== 3. VISUALISER DES ÉCHANTILLONS ==========
        print("\n🖼️  ÉTAPE 3/6: Aperçu du dataset")
        print("-" * 60)
        visualize_dataset_samples(full_dataset, num_samples=3)
        
        # ========== 4. CRÉER LE MODÈLE ==========
        print("\n🧠 ÉTAPE 4/6: Création du modèle")
        print("-" * 60)
        
        model = VideoGPT(
            embed_dim=CONFIG['embed_dim'],
            num_heads=CONFIG['num_heads'],
            num_layers=CONFIG['num_layers'],
            img_size=CONFIG['img_size']
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✅ Modèle créé:")
        print(f"   - Paramètres totaux: {total_params:,}")
        print(f"   - Paramètres entraînables: {trainable_params:,}")
        print(f"   - Taille estimée: {total_params * 4 / 1024 / 1024:.2f} MB")
        
        # Test forward pass
        print(f"\n🔍 Test du modèle...")
        test_input, test_target = next(iter(train_loader))
        print(f"   - Input shape: {test_input.shape}")
        print(f"   - Target shape: {test_target.shape}")
        
        model_test = model.to(DEVICE)
        test_input_device = test_input.to(DEVICE)
        with torch.no_grad():
            test_output = model_test(test_input_device)
        print(f"   - Output shape: {test_output.shape}")
        print(f"   ✅ Forward pass réussi!")
        
        # ========== 5. ENTRAÎNER ==========
        print("\n🏋️  ÉTAPE 5/6: Entraînement")
        print("-" * 60)
        
        history = train_model(
            model,
            train_loader,
            val_loader,
            num_epochs=CONFIG['num_epochs'],
            lr=CONFIG['learning_rate'],
            device=DEVICE,
            visualize_every=CONFIG['visualize_every']
        )
        
        # Sauvegarder le modèle final
        torch.save(model.state_dict(), 'video_gpt_final.pth')
        print("\n✅ Modèle final sauvegardé: video_gpt_final.pth")
        
        # Courbe d'apprentissage
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_loss'], label='Train Loss', marker='o')
        if history['val_loss']:
            plt.plot(history['val_loss'], label='Validation Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Courbe d\'apprentissage')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('training_curve.png', dpi=150, bbox_inches='tight')
        print("✅ Courbe d'apprentissage sauvegardée: training_curve.png")
        plt.close()
        
        # ========== 6. TESTER ==========
        print("\n🎯 ÉTAPE 6/6: Test de prédiction")
        print("-" * 60)
        
        # Test 1: Prédiction simple
        input_frames, target_frame = full_dataset[0]
        predicted_frame = predict_next_frame(model, input_frames, device=DEVICE)
        visualize_prediction(input_frames, target_frame, predicted_frame, "test_prediction.png")
        print("✅ Prédiction simple sauvegardée: test_prediction.png")
        
        # Test 2: Prédiction de séquence
        predictions = predict_sequence(model, input_frames, num_predictions=5, device=DEVICE)
        visualize_sequence_prediction(input_frames, predictions, "test_sequence.png")
        
        print("\n" + "="*60)
        print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
        print("="*60)
        print("\n📁 Fichiers générés:")
        print("   - video_gpt_final.pth (modèle final)")
        print("   - checkpoints/best_model.pth (meilleur modèle)")
        print("   - checkpoints/predictions/epoch_*.png (prédictions par epoch)")
        print("   - training_curve.png (courbe d'apprentissage)")
        print("   - dataset_preview.png (aperçu du dataset)")
        print("   - test_prediction.png (test de prédiction)")
        print("   - test_sequence.png (test de séquence)")
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE:")
        print(f"{'='*60}")
        print(f"{e}")
        print(f"{'='*60}")
        traceback.print_exc()
        print(f"\n💡 Conseils de débogage:")
        print("   1. Vérifiez que le chemin du dataset est correct")
        print("   2. Réduisez img_size si vous manquez de mémoire GPU")
        print("   3. Réduisez batch_size si vous manquez de mémoire")
        print("   4. Vérifiez que CUDA est disponible si vous utilisez GPU")

# ==================== 9. FONCTION POUR CHARGER UN MODÈLE ==========
def load_model(checkpoint_path, config, device='cuda'):
    """Charge un modèle sauvegardé"""
    model = VideoGPT(
        embed_dim=config.get('embed_dim', 256),
        num_heads=config.get('num_heads', 8),
        num_layers=config.get('num_layers', 4),
        img_size=config.get('img_size', 64)
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"✅ Modèle chargé depuis {checkpoint_path}")
    return model

# ==================== POINT D'ENTRÉE ==========
if __name__ == "__main__":
    main()