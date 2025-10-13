import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw
import cv2
import os
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque

# Configuration pour optimiser la mémoire
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

class DrawingEnvironment:
    """Environnement de dessin où l'IA apprend à dessiner trait par trait"""
    
    def __init__(self, canvas_size=256, max_strokes=500):
        self.canvas_size = canvas_size
        self.max_strokes = max_strokes
        self.reset()
        
    def reset(self):
        """Réinitialise le canvas"""
        self.canvas = np.ones((self.canvas_size, self.canvas_size, 3), dtype=np.float32)
        self.stroke_count = 0
        self.pen_position = [self.canvas_size // 2, self.canvas_size // 2]
        return self.canvas.copy()
    
    def draw_stroke(self, action):
        """
        Dessine un trait basé sur l'action
        action: [x_start, y_start, x_end, y_end, thickness, r, g, b]
        """
        x1, y1, x2, y2, thickness, r, g, b = action
        
        # Convertir les coordonnées normalisées en pixels
        x1 = int(x1 * self.canvas_size)
        y1 = int(y1 * self.canvas_size)
        x2 = int(x2 * self.canvas_size)
        y2 = int(y2 * self.canvas_size)
        thickness = max(1, int(thickness * 10))
        
        # Créer une image PIL pour dessiner
        img = Image.fromarray((self.canvas * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        
        # Dessiner le trait
        color = (int(r * 255), int(g * 255), int(b * 255))
        draw.line([(x1, y1), (x2, y2)], fill=color, width=thickness)
        
        # Mettre à jour le canvas
        self.canvas = np.array(img) / 255.0
        self.stroke_count += 1
        self.pen_position = [x2, y2]
        
        return self.canvas.copy()
    
    def draw_bezier_curve(self, action):
        """Dessine une courbe de Bézier pour des traits plus fluides"""
        x1, y1, cx, cy, x2, y2, thickness, r, g, b = action
        
        # Convertir en pixels
        points = []
        for t in np.linspace(0, 1, 20):
            x = (1-t)**2 * x1 + 2*(1-t)*t * cx + t**2 * x2
            y = (1-t)**2 * y1 + 2*(1-t)*t * cy + t**2 * y2
            points.append((int(x * self.canvas_size), int(y * self.canvas_size)))
        
        # Dessiner la courbe
        img = Image.fromarray((self.canvas * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        color = (int(r * 255), int(g * 255), int(b * 255))
        thickness = max(1, int(thickness * 10))
        
        for i in range(len(points) - 1):
            draw.line([points[i], points[i+1]], fill=color, width=thickness)
        
        self.canvas = np.array(img) / 255.0
        self.stroke_count += 1
        
        return self.canvas.copy()

class StrokePredictor(nn.Module):
    """Réseau de neurones qui prédit le prochain trait à dessiner"""
    
    def __init__(self, hidden_dim=512):
        super(StrokePredictor, self).__init__()
        
        # Encodeur CNN pour analyser l'état actuel du canvas et l'image cible
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 32, 3, stride=2, padding=1),  # 6 canaux: 3 canvas + 3 target
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # LSTM pour la mémoire des traits précédents
        self.lstm = nn.LSTM(128 * 8 * 8, hidden_dim, num_layers=2, batch_first=True)
        
        # Décodeur pour prédire les paramètres du prochain trait
        self.stroke_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # 10 paramètres pour une courbe de Bézier
        )
        
        # Tête pour décider si continuer à dessiner
        self.stop_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, canvas, target, hidden=None):
        # Concaténer canvas actuel et image cible
        x = torch.cat([canvas, target], dim=1)
        
        # Encoder
        features = self.encoder(x)
        features = features.view(features.size(0), 1, -1)
        
        # LSTM
        lstm_out, hidden = self.lstm(features, hidden)
        lstm_out = lstm_out[:, -1, :]
        
        # Prédire le prochain trait
        stroke_params = self.stroke_decoder(lstm_out)
        stroke_params = torch.sigmoid(stroke_params)  # Normaliser entre 0 et 1
        
        # Prédire si on doit arrêter
        stop_prob = self.stop_head(lstm_out)
        
        return stroke_params, stop_prob, hidden

class AnimeDataset(Dataset):
    """Dataset pour charger les images d'animé"""
    
    def __init__(self, root_dir, image_size=256):
        self.root_dir = root_dir
        self.image_size = image_size
        self.images = []
        
        # Charger toutes les images
        for filename in os.listdir(root_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                self.images.append(os.path.join(root_dir, filename))
        
        print(f"Dataset chargé: {len(self.images)} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        image = np.array(image) / 255.0
        return torch.FloatTensor(image).permute(2, 0, 1)

class DrawingAgent:
    """Agent qui apprend à dessiner"""
    
    def __init__(self, device='cuda', lr=1e-4):
        self.device = device
        self.env = DrawingEnvironment()
        self.model = StrokePredictor().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)
        
    def draw_image(self, target_image, max_strokes=300, training=False):
        """Dessine une image trait par trait"""
        canvas = self.env.reset()
        strokes = []
        hidden = None
        
        canvas_tensor = torch.FloatTensor(canvas).permute(2, 0, 1).unsqueeze(0).to(self.device)
        target_tensor = target_image.unsqueeze(0).to(self.device) if target_image.dim() == 3 else target_image
        
        for step in range(max_strokes):
            # Prédire le prochain trait
            with torch.set_grad_enabled(training):
                stroke_params, stop_prob, hidden = self.model(canvas_tensor, target_tensor, hidden)
            
            # Décider si arrêter
            if stop_prob.item() > 0.95 and step > 50:
                break
            
            # Extraire les paramètres du trait
            params = stroke_params[0].cpu().detach().numpy()
            
            # Ajouter du bruit pendant l'entraînement (exploration)
            if training and random.random() < 0.1:
                params += np.random.normal(0, 0.05, params.shape)
                params = np.clip(params, 0, 1)
            
            # Dessiner le trait
            canvas = self.env.draw_bezier_curve(params)
            strokes.append(params)
            
            # Mettre à jour le tensor canvas
            canvas_tensor = torch.FloatTensor(canvas).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            # Détacher le hidden state pour éviter l'accumulation de gradient
            if training and hidden is not None:
                hidden = (hidden[0].detach(), hidden[1].detach())
        
        return canvas, strokes
    
    def compute_loss(self, drawn_image, target_image):
        """Calcule la perte entre l'image dessinée et la cible"""
        drawn_tensor = torch.FloatTensor(drawn_image).permute(2, 0, 1).to(self.device)
        
        # Perte L2 pixel-wise
        l2_loss = F.mse_loss(drawn_tensor, target_image)
        
        # Perte perceptuelle (utilisant les features CNN)
        with torch.no_grad():
            drawn_features = self.model.encoder(drawn_tensor.unsqueeze(0).repeat(1, 2, 1, 1))
            target_features = self.model.encoder(target_image.unsqueeze(0).repeat(1, 2, 1, 1))
        perceptual_loss = F.mse_loss(drawn_features, target_features)
        
        # Perte totale
        total_loss = l2_loss + 0.1 * perceptual_loss
        
        return total_loss
    
    def train_step(self, target_image):
        """Un pas d'entraînement"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Dessiner l'image
        drawn_image, strokes = self.draw_image(target_image, training=True)
        
        # Calculer la perte
        loss = self.compute_loss(drawn_image, target_image)
        
        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item(), drawn_image

def train_drawing_ai(dataset_path, epochs=100, batch_size=4):
    """Fonction principale d'entraînement"""
    
    # Vérifier GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation de: {device}")
    
    # Créer le dataset
    dataset = AnimeDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    # Créer l'agent
    agent = DrawingAgent(device=device)
    
    # Entraînement
    losses = []
    
    for epoch in range(epochs):
        epoch_losses = []
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, target_images in enumerate(pbar):
            target_images = target_images.to(device)
            
            batch_loss = 0
            for i in range(target_images.size(0)):
                loss, drawn = agent.train_step(target_images[i])
                batch_loss += loss
                
                # Libérer la mémoire GPU
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
            
            avg_loss = batch_loss / target_images.size(0)
            epoch_losses.append(avg_loss)
            pbar.set_postfix({'Loss': f'{avg_loss:.4f}'})
            
            # Sauvegarder un exemple toutes les 50 batches
            if batch_idx % 50 == 0:
                save_example(target_images[0], drawn, epoch, batch_idx)
        
        # Statistiques de l'epoch
        avg_epoch_loss = np.mean(epoch_losses)
        losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1} - Loss moyenne: {avg_epoch_loss:.4f}")
        
        # Sauvegarder le modèle
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': agent.model.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, f'drawing_ai_epoch_{epoch+1}.pt')
    
    return agent, losses

def save_example(target, drawn, epoch, batch):
    """Sauvegarde des exemples de dessin"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Image cible
    target_img = target.cpu().permute(1, 2, 0).numpy()
    axes[0].imshow(target_img)
    axes[0].set_title('Cible')
    axes[0].axis('off')
    
    # Image dessinée
    axes[1].imshow(drawn)
    axes[1].set_title('Dessinée par l\'IA')
    axes[1].axis('off')
    
    plt.savefig(f'examples/epoch_{epoch}_batch_{batch}.png')
    plt.close()

def generate_new_drawing(agent, prompt_image=None):
    """Génère un nouveau dessin"""
    agent.model.eval()
    
    if prompt_image is None:
        # Créer une image aléatoire comme cible
        prompt_image = torch.rand(3, 256, 256).to(agent.device)
    
    with torch.no_grad():
        drawn_image, strokes = agent.draw_image(prompt_image, training=False)
    
    return drawn_image, strokes

# Fonction d'utilisation
def main():
    # Créer les dossiers nécessaires
    os.makedirs('examples', exist_ok=True)
    
    # Chemin vers votre dataset d'images d'animé
    DATASET_PATH = "/kaggle/input/anima-s-dataset/test"  # Remplacez par votre chemin
    
    # Entraîner l'IA
    print("Démarrage de l'entraînement de l'IA de dessin...")
    agent, losses = train_drawing_ai(DATASET_PATH, epochs=50, batch_size=2)
    
    # Afficher la courbe de perte
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Évolution de la perte pendant l\'entraînement')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('training_loss.png')
    plt.show()
    
    print("Entraînement terminé!")
    
    # Générer quelques exemples
    print("Génération d'exemples...")
    for i in range(5):
        drawn, _ = generate_new_drawing(agent)
        plt.figure(figsize=(8, 8))
        plt.imshow(drawn)
        plt.title(f'Dessin généré #{i+1}')
        plt.axis('off')
        plt.savefig(f'generated_{i+1}.png')
        plt.show()

if __name__ == "__main__":
    main()