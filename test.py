import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

# ==================== Configuration ====================
class Config:
    # Chemins
    dataset_path = "/kaggle/input/anima-s-dataset/test/"  # Changez ceci vers votre dossier
    checkpoint_dir = "checkpoints"
    
    # Hyperparamètres
    image_size = 256  # Taille des images (128x128)
    sequence_length = 2  # Nombre d'images en entrée pour prédire la suivante
    latent_dim = 512  # Dimension de l'espace latent
    hidden_dim = 1024  # Dimension cachée du LSTM
    
    # Entraînement
    batch_size = 8
    num_epochs = 100
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Sauvegarde
    save_every = 10  # Sauvegarder tous les N epochs
    
config = Config()

# ==================== Dataset ====================
class FrameSequenceDataset(Dataset):
    """Dataset pour charger des séquences d'images."""
    
    def __init__(self, root_dir, sequence_length=2, transform=None):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Lister et trier toutes les images
        self.frames = sorted([f for f in os.listdir(root_dir) 
                             if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Vérifier qu'on a assez d'images
        if len(self.frames) < sequence_length + 1:
            raise ValueError(f"Pas assez d'images. Minimum requis: {sequence_length + 1}")
        
        print(f"Dataset initialisé avec {len(self.frames)} images")
        
    def __len__(self):
        # On peut créer (nombre_total - sequence_length) séquences
        return len(self.frames) - self.sequence_length
    
    def __getitem__(self, idx):
        # Charger une séquence d'images
        input_sequence = []
        
        # Charger les images d'entrée
        for i in range(self.sequence_length):
            img_path = os.path.join(self.root_dir, self.frames[idx + i])
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            input_sequence.append(image)
        
        # Charger l'image cible (suivante)
        target_path = os.path.join(self.root_dir, self.frames[idx + self.sequence_length])
        target_image = Image.open(target_path).convert('RGB')
        
        if self.transform:
            target_image = self.transform(target_image)
        
        # Stack les images d'entrée
        input_tensor = torch.stack(input_sequence)
        
        return input_tensor, target_image

# ==================== Modèle ====================
class ImageEncoder(nn.Module):
    """Encodeur CNN pour extraire les features des images."""
    
    def __init__(self, latent_dim=512):
        super(ImageEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            # Input: 3 x 128 x 128
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 64 x 64 x 64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128 x 32 x 32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 256 x 16 x 16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 512 x 8 x 8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),  # 1024 x 4 x 4
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 4 * 4, latent_dim),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x

class ImageDecoder(nn.Module):
    """Décodeur CNN pour reconstruire les images."""
    
    def __init__(self, latent_dim=512):
        super(ImageDecoder, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024 * 4 * 4),
            nn.LeakyReLU(0.2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # 512 x 8 x 8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 256 x 16 x 16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 128 x 32 x 32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64 x 64 x 64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # 3 x 128 x 128
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 1024, 4, 4)
        x = self.decoder(x)
        return x

class FramePredictionModel(nn.Module):
    """Modèle complet pour prédire la prochaine frame."""
    
    def __init__(self, sequence_length=2, latent_dim=512, hidden_dim=1024):
        super(FramePredictionModel, self).__init__()
        
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Encodeur pour extraire les features
        self.encoder = ImageEncoder(latent_dim)
        
        # LSTM pour modéliser la séquence temporelle
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Projection de la sortie LSTM vers l'espace latent
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
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
        
        # Passer par le LSTM
        lstm_out, _ = self.lstm(encoded_sequence)
        
        # Prendre la dernière sortie
        last_hidden = lstm_out[:, -1, :]
        
        # Projeter vers l'espace latent
        predicted_latent = self.projection(last_hidden)
        
        # Décoder pour obtenir l'image prédite
        predicted_frame = self.decoder(predicted_latent)
        
        return predicted_frame

# ==================== Fonctions d'entraînement ====================
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Entraîner le modèle pour une epoch."""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, (input_seq, target) in enumerate(progress_bar):
        input_seq = input_seq.to(device)
        target = target.to(device)
        
        # Forward pass
        predicted = model(input_seq)
        
        # Calculer la loss
        loss = criterion(predicted, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Mise à jour de la barre de progression
        progress_bar.set_postfix({'Loss': loss.item()})
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    """Valider le modèle."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for input_seq, target in tqdm(dataloader, desc="Validation"):
            input_seq = input_seq.to(device)
            target = target.to(device)
            
            predicted = model(input_seq)
            loss = criterion(predicted, target)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def save_checkpoint(model, optimizer, epoch, loss, path):
    """Sauvegarder un checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"Checkpoint sauvegardé: {path}")

def load_checkpoint(model, optimizer, path):
    """Charger un checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss

def visualize_prediction(model, dataset, device, num_samples=3):
    """Visualiser les prédictions du modèle."""
    model.eval()
    
    fig, axes = plt.subplots(num_samples, config.sequence_length + 2, figsize=(15, 5 * num_samples))
    
    with torch.no_grad():
        for i in range(num_samples):
            # Obtenir un échantillon aléatoire
            idx = np.random.randint(0, len(dataset))
            input_seq, target = dataset[idx]
            
            # Préparer pour le modèle
            input_batch = input_seq.unsqueeze(0).to(device)
            
            # Prédire
            predicted = model(input_batch)
            
            # Afficher les images d'entrée
            for j in range(config.sequence_length):
                img = input_seq[j].cpu().numpy().transpose(1, 2, 0)
                img = (img + 1) / 2  # Dénormaliser de [-1, 1] à [0, 1]
                axes[i, j].imshow(img)
                axes[i, j].set_title(f"Input {j+1}")
                axes[i, j].axis('off')
            
            # Afficher la vraie image suivante
            target_img = target.cpu().numpy().transpose(1, 2, 0)
            target_img = (target_img + 1) / 2
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
    plt.savefig('predictions.png')
    plt.show()

# ==================== Fonction principale ====================
def main():
    # Créer le dossier de checkpoints
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Transformations pour les images
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normaliser à [-1, 1]
    ])
    
    # Créer le dataset
    dataset = FrameSequenceDataset(
        root_dir=config.dataset_path,
        sequence_length=config.sequence_length,
        transform=transform
    )
    
    # Diviser en train/validation (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Créer les dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Créer le modèle
    model = FramePredictionModel(
        sequence_length=config.sequence_length,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim
    ).to(config.device)
    
    print(f"Modèle créé avec {sum(p.numel() for p in model.parameters())} paramètres")
    
    # Définir la loss et l'optimiseur
    criterion = nn.MSELoss()  # On peut aussi utiliser L1Loss ou une combinaison
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Entraînement
    best_val_loss = float('inf')
    
    for epoch in range(1, config.num_epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{config.num_epochs}")
        print(f"{'='*50}")
        
        # Entraîner
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config.device)
        
        # Valider
        val_loss = validate(model, val_loader, criterion, config.device)
        
        # Ajuster le learning rate
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Sauvegarder le meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(config.checkpoint_dir, 'best_model.pth')
            )
        
        # Sauvegarder périodiquement
        if epoch % config.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            )
            
            # Visualiser les prédictions
            visualize_prediction(model, val_dataset, config.device)
    
    print("\nEntraînement terminé!")
    
    # Charger le meilleur modèle pour la prédiction finale
    epoch, _ = load_checkpoint(
        model, optimizer,
        os.path.join(config.checkpoint_dir, 'best_model.pth')
    )
    print(f"Meilleur modèle chargé (epoch {epoch})")
    
    # Visualisation finale
    visualize_prediction(model, val_dataset, config.device, num_samples=5)

# ==================== Fonction de prédiction ====================
def predict_next_frame(model, frame_paths, device, transform):
    """
    Prédire la prochaine frame à partir d'une séquence d'images.
    
    Args:
        model: Le modèle entraîné
        frame_paths: Liste des chemins vers les frames d'entrée
        device: Device (CPU/GPU)
        transform: Transformations à appliquer
    
    Returns:
        Image prédite (PIL Image)
    """
    model.eval()
    
    # Charger et transformer les images
    frames = []
    for path in frame_paths:
        img = Image.open(path).convert('RGB')
        if transform:
            img = transform(img)
        frames.append(img)
    
    # Stack et ajouter la dimension batch
    input_tensor = torch.stack(frames).unsqueeze(0).to(device)
    
    # Prédire
    with torch.no_grad():
        predicted = model(input_tensor)
    
    # Convertir en image PIL
    pred_img = predicted[0].cpu().numpy().transpose(1, 2, 0)
    pred_img = (pred_img + 1) / 2  # Dénormaliser
    pred_img = np.clip(pred_img, 0, 1)
    pred_img = (pred_img * 255).astype(np.uint8)
    
    return Image.fromarray(pred_img)

# ==================== Exemple d'utilisation pour la prédiction ====================
def demo_prediction():
    """Démonstration de prédiction sur de nouvelles frames."""
    
    # Charger le modèle
    model = FramePredictionModel(
        sequence_length=config.sequence_length,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim
    ).to(config.device)
    
    # Charger les poids
    checkpoint = torch.load(os.path.join(config.checkpoint_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Transformations
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Exemple: prédire à partir de frame_0001.png et frame_0002.png
    frame_paths = [
        os.path.join(config.dataset_path, "frame_0001.png"),
        os.path.join(config.dataset_path, "frame_0002.png")
    ]
    
    # Prédire
    predicted_frame = predict_next_frame(model, frame_paths, config.device, transform)
    
    # Sauvegarder la prédiction
    predicted_frame.save("predicted_frame_0003.png")
    print("Prédiction sauvegardée: predicted_frame_0003.png")
    
    # Afficher
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for i, path in enumerate(frame_paths):
        img = Image.open(path)
        axes[i].imshow(img)
        axes[i].set_title(f"Input Frame {i+1}")
        axes[i].axis('off')
    
    axes[2].imshow(predicted_frame)
    axes[2].set_title("Predicted Next Frame")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Pour entraîner le modèle
    main()
    
    # Pour tester la prédiction après l'entraînement
    # demo_prediction()