import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from datetime import datetime

# ==================== Configuration ====================
class Config:
    # Chemins
    FRAMES_DIR = "/kaggle/input/anima-s-dataset/test"  # Dossier contenant les frames
    CHECKPOINT_DIR = "checkpoints"
    OUTPUT_DIR = "predictions"
    
    # Hyperparamètres
    SEQUENCE_LENGTH = 5  # Nombre de frames en entrée pour prédire la suivante
    IMG_SIZE = (256, 256)  # Taille des images
    BATCH_SIZE = 8
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model
    HIDDEN_DIM = 512
    NUM_LAYERS = 3

# ==================== Dataset ====================
class AnimeFrameDataset(Dataset):
    """Dataset pour charger les séquences de frames d'animation"""
    
    def __init__(self, frames_dir, sequence_length=5, transform=None):
        self.frames_dir = Path(frames_dir)
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Récupérer toutes les frames triées
        self.frame_paths = sorted(self.frames_dir.glob("frame_*.png"))
        
        # Calculer le nombre de séquences possibles
        self.num_sequences = len(self.frame_paths) - sequence_length
        
        if self.num_sequences <= 0:
            raise ValueError(f"Pas assez de frames. Minimum requis: {sequence_length + 1}")
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        # Charger une séquence de frames
        input_frames = []
        for i in range(self.sequence_length):
            frame_path = self.frame_paths[idx + i]
            frame = Image.open(frame_path).convert('RGB')
            
            if self.transform:
                frame = self.transform(frame)
            input_frames.append(frame)
        
        # La frame cible est la suivante dans la séquence
        target_frame_path = self.frame_paths[idx + self.sequence_length]
        target_frame = Image.open(target_frame_path).convert('RGB')
        
        if self.transform:
            target_frame = self.transform(target_frame)
        
        # Stack les frames d'entrée
        input_tensor = torch.stack(input_frames)
        
        return input_tensor, target_frame

# ==================== Modèle ====================
class ConvLSTMCell(nn.Module):
    """Cellule ConvLSTM pour traiter les séquences vidéo"""
    
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        
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

class FramePredictionModel(nn.Module):
    """Modèle principal pour la prédiction de frames"""
    
    def __init__(self, input_channels=3, hidden_dim=64, num_layers=3):
        super(FramePredictionModel, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # ConvLSTM layers
        self.convlstm_layers = nn.ModuleList([
            ConvLSTMCell(
                input_dim=256 if i == 0 else hidden_dim,
                hidden_dim=hidden_dim,
                kernel_size=3
            )
            for i in range(num_layers)
        ])
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, input_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Pour avoir des valeurs entre 0 et 1
        )
    
    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size()
        
        # Initialiser les états cachés
        h = [None] * self.num_layers
        c = [None] * self.num_layers
        
        # Traiter chaque frame de la séquence
        for t in range(seq_len):
            # Encoder la frame actuelle
            encoded = self.encoder(x[:, t])
            
            # Passer par les couches ConvLSTM
            for layer_idx, convlstm in enumerate(self.convlstm_layers):
                if h[layer_idx] is None:
                    h_shape = (batch_size, self.hidden_dim, 
                              encoded.size(2), encoded.size(3))
                    h[layer_idx] = torch.zeros(h_shape, device=x.device)
                    c[layer_idx] = torch.zeros(h_shape, device=x.device)
                
                if layer_idx == 0:
                    h[layer_idx], c[layer_idx] = convlstm(
                        encoded, (h[layer_idx], c[layer_idx])
                    )
                else:
                    h[layer_idx], c[layer_idx] = convlstm(
                        h[layer_idx-1], (h[layer_idx], c[layer_idx])
                    )
        
        # Decoder pour générer la frame suivante
        output = self.decoder(h[-1])
        
        return output

# ==================== Fonctions d'entraînement ====================
class Trainer:
    def __init__(self, model, config):
        self.model = model.to(config.DEVICE)
        self.config = config
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Créer les dossiers nécessaires
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, dataloader):
        self.model.train()
        epoch_loss = 0
        
        with tqdm(dataloader, desc="Training") as pbar:
            for batch_idx, (input_frames, target_frame) in enumerate(pbar):
                input_frames = input_frames.to(self.config.DEVICE)
                target_frame = target_frame.to(self.config.DEVICE)
                
                # Forward pass
                self.optimizer.zero_grad()
                predicted_frame = self.model(input_frames)
                
                # Calculer la loss
                loss = self.criterion(predicted_frame, target_frame)
                
                # Ajouter une loss perceptuelle (optionnel)
                perceptual_loss = self.perceptual_loss(predicted_frame, target_frame)
                total_loss = loss + 0.1 * perceptual_loss
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        return epoch_loss / len(dataloader)
    
    def perceptual_loss(self, pred, target):
        """Loss perceptuelle basée sur les features"""
        # Simplifiée ici, vous pouvez utiliser VGG pour une vraie loss perceptuelle
        return torch.mean(torch.abs(pred - target))
    
    def validate(self, dataloader):
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for input_frames, target_frame in dataloader:
                input_frames = input_frames.to(self.config.DEVICE)
                target_frame = target_frame.to(self.config.DEVICE)
                
                predicted_frame = self.model(input_frames)
                loss = self.criterion(predicted_frame, target_frame)
                val_loss += loss.item()
        
        return val_loss / len(dataloader)
    
    def save_checkpoint(self, epoch, loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        checkpoint_path = os.path.join(
            self.config.CHECKPOINT_DIR, 
            f'checkpoint_epoch_{epoch}_loss_{loss:.4f}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        print(f"✅ Checkpoint sauvegardé: {checkpoint_path}")
    
    def train(self, train_loader, val_loader=None):
        print(f"🚀 Début de l'entraînement sur {self.config.DEVICE}")
        
        best_loss = float('inf')
        
        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\n📊 Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
            
            # Entraînement
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            print(f"📉 Train Loss: {train_loss:.4f}")
            
            # Validation
            if val_loader:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                print(f"📈 Val Loss: {val_loss:.4f}")
                self.scheduler.step(val_loss)
                current_loss = val_loss
            else:
                current_loss = train_loss
            
            # Sauvegarder le meilleur modèle
            if current_loss < best_loss:
                best_loss = current_loss
                self.save_checkpoint(epoch, current_loss)
            
            # Visualiser les prédictions toutes les 10 epochs
            if (epoch + 1) % 10 == 0:
                self.visualize_predictions(train_loader, epoch)
    
    def visualize_predictions(self, dataloader, epoch):
        """Visualiser les prédictions du modèle"""
        self.model.eval()
        
        with torch.no_grad():
            # Prendre un batch
            input_frames, target_frame = next(iter(dataloader))
            input_frames = input_frames.to(self.config.DEVICE)
            target_frame = target_frame.to(self.config.DEVICE)
            
            # Prédire
            predicted_frame = self.model(input_frames)
            
            # Convertir en numpy pour visualisation
            input_last = input_frames[0, -1].cpu().numpy().transpose(1, 2, 0)
            target = target_frame[0].cpu().numpy().transpose(1, 2, 0)
            predicted = predicted_frame[0].cpu().numpy().transpose(1, 2, 0)
            
            # Créer la figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(input_last)
            axes[0].set_title("Dernière frame d'entrée")
            axes[0].axis('off')
            
            axes[1].imshow(target)
            axes[1].set_title("Frame cible (vérité)")
            axes[1].axis('off')
            
            axes[2].imshow(predicted)
            axes[2].set_title("Frame prédite")
            axes[2].axis('off')
            
            plt.suptitle(f"Epoch {epoch + 1}")
            plt.tight_layout()
            
            # Sauvegarder
            save_path = os.path.join(
                self.config.OUTPUT_DIR, 
                f'prediction_epoch_{epoch+1}.png'
            )
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            print(f"📸 Prédiction sauvegardée: {save_path}")

# ==================== Générateur de frames ====================
class FrameGenerator:
    """Classe pour générer de nouvelles frames"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.transform = transforms.Compose([
            transforms.Resize(config.IMG_SIZE),
            transforms.ToTensor()
        ])
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.config.DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"✅ Modèle chargé depuis: {checkpoint_path}")
    
    def generate_next_frame(self, input_frames_paths):
        """Générer la frame suivante à partir d'une séquence"""
        frames = []
        
        for path in input_frames_paths:
            frame = Image.open(path).convert('RGB')
            frame = self.transform(frame)
            frames.append(frame)
        
        # Stack et ajouter batch dimension
        input_tensor = torch.stack(frames).unsqueeze(0).to(self.config.DEVICE)
        
        with torch.no_grad():
            predicted_frame = self.model(input_tensor)
        
        # Convertir en image
        predicted_np = predicted_frame[0].cpu().numpy().transpose(1, 2, 0)
        predicted_np = (predicted_np * 255).astype(np.uint8)
        
        return predicted_np
    
    def generate_sequence(self, initial_frames_paths, num_frames=30):
        """Générer une séquence de frames"""
        generated_frames = []
        current_frames = list(initial_frames_paths)
        
        for i in range(num_frames):
            print(f"Génération frame {i+1}/{num_frames}")
            
            # Générer la prochaine frame
            next_frame = self.generate_next_frame(current_frames[-self.config.SEQUENCE_LENGTH:])
            generated_frames.append(next_frame)
            
            # Sauvegarder la frame
            save_path = os.path.join(
                self.config.OUTPUT_DIR,
                f'generated_frame_{i:04d}.png'
            )
            cv2.imwrite(save_path, cv2.cvtColor(next_frame, cv2.COLOR_RGB2BGR))
            
            # Mettre à jour la séquence pour la prochaine prédiction
            # (Utiliser la frame générée comme entrée)
            temp_path = f"temp_frame_{i}.png"
            cv2.imwrite(temp_path, cv2.cvtColor(next_frame, cv2.COLOR_RGB2BGR))
            current_frames.append(temp_path)
            
            # Nettoyer les fichiers temporaires
            if i > 0:
                os.remove(f"temp_frame_{i-1}.png")
        
        return generated_frames
    
    def create_video(self, frames, output_path='generated_animation.mp4', fps=24):
        """Créer une vidéo à partir des frames générées"""
        if not frames:
            print("❌ Aucune frame à convertir en vidéo")
            return
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        video_writer.release()
        print(f"✅ Vidéo créée: {output_path}")

# ==================== Fonction principale ====================
def main():
    # Configuration
    config = Config()
    
    # Transformations
    transform = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.ToTensor()
    ])
    
    # Dataset et DataLoader
    print("📁 Chargement du dataset...")
    dataset = AnimeFrameDataset(
        frames_dir=config.FRAMES_DIR,
        sequence_length=config.SEQUENCE_LENGTH,
        transform=transform
    )
    
    # Diviser en train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=2
    )
    
    print(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Modèle
    print("🤖 Création du modèle...")
    model = FramePredictionModel(
        input_channels=3,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS
    )
    
    # Entraînement
    trainer = Trainer(model, config)
    trainer.train(train_loader, val_loader)
    
    print("\n✅ Entraînement terminé!")
    
    # Générer des nouvelles frames
    print("\n🎬 Génération de nouvelles frames...")
    generator = FrameGenerator(model, config)
    
    # Charger le meilleur checkpoint
    checkpoints = sorted(Path(config.CHECKPOINT_DIR).glob("*.pth"))
    if checkpoints:
        generator.load_checkpoint(str(checkpoints[-1]))
        
        # Prendre les premières frames comme seed
        initial_frames = dataset.frame_paths[:config.SEQUENCE_LENGTH]
        initial_frames_paths = [str(p) for p in initial_frames]
        
        # Générer une séquence
        generated_frames = generator.generate_sequence(
            initial_frames_paths, 
            num_frames=30
        )
        
        # Créer une vidéo
        generator.create_video(generated_frames)
    
    # Plot les courbes de loss
    plt.figure(figsize=(10, 5))
    plt.plot(trainer.train_losses, label='Train Loss')
    if trainer.val_losses:
        plt.plot(trainer.val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Courbes d\'apprentissage')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curves.png')
    plt.show()

# ==================== Script d'extraction de frames ====================
def extract_frames_from_video(video_path, output_dir, skip_frames=1):
    """
    Extraire les frames d'une vidéo d'animation
    
    Args:
        video_path: Chemin vers la vidéo
        output_dir: Dossier de sortie pour les frames
        skip_frames: Nombre de frames à ignorer entre chaque extraction
    """
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
        
        frame_count += 1
    
    cap.release()
    print(f"✅ {saved_count} frames extraites vers {output_dir}")

if __name__ == "__main__":
    # Option 1: Extraire les frames d'une vidéo
    # extract_frames_from_video('anime_episode.mp4', 'anime_frames', skip_frames=2)
    
    # Option 2: Entraîner le modèle
    main()