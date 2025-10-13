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

# ==================== 1. GÉNÉRATION DE DONNÉES SYNTHÉTIQUES ====================
def generate_synthetic_data(num_sequences=1000, frames_per_seq=5, img_size=64, save_dir="data/frames"):
    """
    Génère des séquences d'images synthétiques (balle qui bouge)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for seq_idx in tqdm(range(num_sequences), desc="Génération des données"):
        # Position et vitesse aléatoires
        x, y = np.random.randint(10, img_size-10), np.random.randint(10, img_size-10)
        vx, vy = np.random.randint(-5, 6), np.random.randint(-5, 6)
        color = np.random.rand(3)
        
        seq_dir = os.path.join(save_dir, f"seq_{seq_idx:04d}")
        os.makedirs(seq_dir, exist_ok=True)
        
        for frame_idx in range(frames_per_seq):
            # Créer l'image
            img = np.zeros((img_size, img_size, 3))
            
            # Dessiner un cercle
            y_grid, x_grid = np.ogrid[:img_size, :img_size]
            mask = (x_grid - x)**2 + (y_grid - y)**2 <= 25
            img[mask] = color
            
            # Sauvegarder
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
            img_pil.save(os.path.join(seq_dir, f"frame_{frame_idx:04d}.png"))
            
            # Mettre à jour la position
            x, y = x + vx, y + vy
            
            # Rebondir sur les bords
            if x <= 5 or x >= img_size - 5:
                vx = -vx
            if y <= 5 or y >= img_size - 5:
                vy = -vy
            
            x = np.clip(x, 5, img_size - 5)
            y = np.clip(y, 5, img_size - 5)

# ==================== 2. DATASET ====================
class VideoSequenceDataset(Dataset):
    """
    Dataset pour les séquences d'images
    """
    def __init__(self, data_dir, context_length=2, img_size=64):
        self.data_dir = data_dir
        self.context_length = context_length
        self.img_size = img_size
        
        # Trouver toutes les séquences
        self.sequences = sorted([d for d in Path(data_dir).iterdir() if d.is_dir()])
        self.samples = []
        
        for seq_dir in self.sequences:
            frames = sorted(list(seq_dir.glob("*.png")))
            # Créer des échantillons avec context_length images en entrée + 1 en sortie
            for i in range(len(frames) - context_length):
                self.samples.append({
                    'input_frames': frames[i:i+context_length],
                    'target_frame': frames[i+context_length]
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Charger les images d'entrée
        input_imgs = []
        for frame_path in sample['input_frames']:
            img = Image.open(frame_path).resize((self.img_size, self.img_size))
            img = np.array(img).astype(np.float32) / 255.0
            input_imgs.append(img)
        
        # Charger l'image cible
        target_img = Image.open(sample['target_frame']).resize((self.img_size, self.img_size))
        target_img = np.array(target_img).astype(np.float32) / 255.0
        
        # Convertir en tenseurs (B, C, H, W)
        input_imgs = torch.FloatTensor(np.array(input_imgs)).permute(0, 3, 1, 2)
        target_img = torch.FloatTensor(target_img).permute(2, 0, 1)
        
        return input_imgs, target_img

# ==================== 3. MODÈLE DE TOKENISATION (VQ-VAE simplifié) ====================
class ImageTokenizer(nn.Module):
    """
    Encode les images en tokens discrets (comme un tokenizer de texte)
    """
    def __init__(self, img_size=64, patch_size=8, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Encoder : Image -> Embeddings
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),  # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(256, embed_dim, 3, 1, 1),
        )
        
        # Decoder : Embeddings -> Image
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),  # 32x32 -> 64x64
            nn.Sigmoid()
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

# ==================== 4. MODÈLE TRANSFORMER (type GPT) ====================
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
    """
    Modèle type GPT pour prédire la prochaine image
    """
    def __init__(self, embed_dim=256, num_heads=8, num_layers=6, img_size=64):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Tokenizer d'images
        self.tokenizer = ImageTokenizer(img_size=img_size, embed_dim=embed_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim * 64)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim * 64, 
            nhead=num_heads,
            dim_feedforward=embed_dim * 128,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Projection pour décoder
        self.projection = nn.Linear(embed_dim * 64, embed_dim * 64)
    
    def forward(self, input_frames):
        """
        input_frames: (batch, num_frames, 3, H, W)
        """
        batch_size, num_frames, C, H, W = input_frames.shape
        
        # Encoder chaque frame
        encoded_frames = []
        for i in range(num_frames):
            _, z = self.tokenizer(input_frames[:, i])  # (B, embed_dim, 8, 8)
            z = z.flatten(1)  # (B, embed_dim * 64)
            encoded_frames.append(z)
        
        # Stack les frames encodés
        encoded_seq = torch.stack(encoded_frames, dim=1)  # (B, num_frames, embed_dim*64)
        
        # Positional encoding
        encoded_seq = self.pos_encoder(encoded_seq)
        
        # Transformer
        transformed = self.transformer(encoded_seq)  # (B, num_frames, embed_dim*64)
        
        # Prendre la dernière sortie pour prédire la prochaine frame
        next_frame_encoding = self.projection(transformed[:, -1])  # (B, embed_dim*64)
        
        # Reshape et décoder
        next_frame_encoding = next_frame_encoding.view(batch_size, self.embed_dim, 8, 8)
        predicted_frame = self.tokenizer.decode(next_frame_encoding)
        
        return predicted_frame

# ==================== 5. ENTRAÎNEMENT ====================
def train_model(model, train_loader, num_epochs=50, lr=1e-4, device='cuda'):
    """
    Entraîne le modèle de prédiction d'images
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    history = {'train_loss': []}
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for input_frames, target_frame in pbar:
            input_frames = input_frames.to(device)
            target_frame = target_frame.to(device)
            
            # Forward pass
            predicted_frame = model(input_frames)
            loss = criterion(predicted_frame, target_frame)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.6f}")
    
    return history

# ==================== 6. PRÉDICTION ET VISUALISATION ====================
def predict_next_frame(model, input_frames, device='cuda'):
    """
    Prédit la prochaine image
    """
    model.eval()
    with torch.no_grad():
        input_frames = input_frames.unsqueeze(0).to(device)
        predicted_frame = model(input_frames)
    return predicted_frame.cpu().squeeze(0)

def visualize_prediction(input_frames, target_frame, predicted_frame):
    """
    Visualise les résultats
    """
    num_inputs = input_frames.shape[0]
    
    fig, axes = plt.subplots(1, num_inputs + 2, figsize=(15, 3))
    
    # Images d'entrée
    for i in range(num_inputs):
        axes[i].imshow(input_frames[i].permute(1, 2, 0).numpy())
        axes[i].set_title(f"Input Frame {i+1}")
        axes[i].axis('off')
    
    # Image cible
    axes[num_inputs].imshow(target_frame.permute(1, 2, 0).numpy())
    axes[num_inputs].set_title("Target (Ground Truth)")
    axes[num_inputs].axis('off')
    
    # Image prédite
    axes[num_inputs + 1].imshow(predicted_frame.permute(1, 2, 0).numpy())
    axes[num_inputs + 1].set_title("Predicted")
    axes[num_inputs + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig("prediction_result.png", dpi=150, bbox_inches='tight')
    plt.show()

# ==================== 7. FONCTION PRINCIPALE ====================
def main():
    # Configuration
    IMG_SIZE = 64
    CONTEXT_LENGTH = 2  # Nombre d'images en entrée
    BATCH_SIZE = 16
    NUM_EPOCHS = 30
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"🚀 Utilisation du device: {DEVICE}")
    
    # 1. Générer les données
    print("\n📊 Génération des données synthétiques...")
    generate_synthetic_data(num_sequences=500, frames_per_seq=5, img_size=IMG_SIZE)
    
    # 2. Créer le dataset
    print("\n📦 Création du dataset...")
    dataset = VideoSequenceDataset("data/frames", context_length=CONTEXT_LENGTH, img_size=IMG_SIZE)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"   Nombre d'échantillons: {len(dataset)}")
    
    # 3. Créer le modèle
    print("\n🧠 Création du modèle VideoGPT...")
    model = VideoGPT(embed_dim=256, num_heads=8, num_layers=4, img_size=IMG_SIZE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Nombre de paramètres: {total_params:,}")
    
    # 4. Entraîner
    print("\n🏋️ Début de l'entraînement...")
    history = train_model(model, train_loader, num_epochs=NUM_EPOCHS, lr=1e-4, device=DEVICE)
    
    # 5. Sauvegarder le modèle
    print("\n💾 Sauvegarde du modèle...")
    torch.save(model.state_dict(), "video_gpt_model.pth")
    
    # 6. Tester la prédiction
    print("\n🎯 Test de prédiction...")
    input_frames, target_frame = dataset[0]
    predicted_frame = predict_next_frame(model, input_frames, device=DEVICE)
    
    # 7. Visualiser
    print("\n📊 Visualisation des résultats...")
    visualize_prediction(input_frames, target_frame, predicted_frame)
    
    # 8. Plot de la courbe d'apprentissage
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'])
    plt.title("Courbe d'apprentissage")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.grid(True)
    plt.savefig("training_curve.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✅ Entraînement terminé!")
    print(f"   Modèle sauvegardé: video_gpt_model.pth")
    print(f"   Résultats sauvegardés: prediction_result.png, training_curve.png")

if __name__ == "__main__":
    main()