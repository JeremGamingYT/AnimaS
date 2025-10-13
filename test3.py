import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2
import os
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque
import torchvision.models as models

# Configuration pour optimiser la mémoire
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

class AdvancedDrawingEnvironment:
    """Environnement de dessin avancé avec primitives variées pour dessiner des animés"""
    
    def __init__(self, canvas_size=256, max_strokes=500):
        self.canvas_size = canvas_size
        self.max_strokes = max_strokes
        self.reset()
        
    def reset(self):
        """Réinitialise le canvas avec des couches"""
        self.canvas = np.ones((self.canvas_size, self.canvas_size, 3), dtype=np.float32)
        self.layers = []  # Stocke les différentes couches
        self.stroke_count = 0
        return self.canvas.copy()
    
    def apply_primitive(self, primitive_type, params):
        """
        Applique une primitive de dessin
        primitive_type: 0=bezier, 1=ellipse, 2=polygon, 3=gradient, 4=texture
        """
        img = Image.fromarray((self.canvas * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img, 'RGBA')
        
        if primitive_type == 0:  # Courbe de Bézier complexe
            self._draw_bezier(draw, params)
        elif primitive_type == 1:  # Ellipse/Cercle (pour les yeux, visages)
            self._draw_ellipse(draw, params)
        elif primitive_type == 2:  # Polygone (pour les cheveux, vêtements)
            self._draw_polygon(draw, params)
        elif primitive_type == 3:  # Gradient (pour les ombres, lumières)
            img = self._apply_gradient(img, params)
        elif primitive_type == 4:  # Texture/Pattern
            img = self._apply_texture(img, params)
        
        # Appliquer un léger flou pour lisser
        if random.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        self.canvas = np.array(img) / 255.0
        self.stroke_count += 1
        return self.canvas.copy()
    
    def _draw_bezier(self, draw, params):
        """Dessine une courbe de Bézier avec gradient de couleur"""
        # params: [x1, y1, cx1, cy1, cx2, cy2, x2, y2, thickness, r1, g1, b1, r2, g2, b2, alpha]
        points = []
        colors = []
        
        for t in np.linspace(0, 1, 30):
            # Points de la courbe cubique
            x = ((1-t)**3 * params[0] + 
                 3*(1-t)**2*t * params[2] + 
                 3*(1-t)*t**2 * params[4] + 
                 t**3 * params[6]) * self.canvas_size
            y = ((1-t)**3 * params[1] + 
                 3*(1-t)**2*t * params[3] + 
                 3*(1-t)*t**2 * params[5] + 
                 t**3 * params[7]) * self.canvas_size
            points.append((int(x), int(y)))
            
            # Interpolation de couleur
            r = int((params[9] * (1-t) + params[12] * t) * 255)
            g = int((params[10] * (1-t) + params[13] * t) * 255)
            b = int((params[11] * (1-t) + params[14] * t) * 255)
            a = int(params[15] * 255)
            colors.append((r, g, b, a))
        
        thickness = max(1, int(params[8] * 15))
        
        # Dessiner avec variation de couleur
        for i in range(len(points) - 1):
            draw.line([points[i], points[i+1]], fill=colors[i], width=thickness)
    
    def _draw_ellipse(self, draw, params):
        """Dessine une ellipse (utile pour les yeux, visages ronds)"""
        # params: [cx, cy, rx, ry, rotation, r, g, b, fill_r, fill_g, fill_b, alpha, fill_alpha]
        cx = int(params[0] * self.canvas_size)
        cy = int(params[1] * self.canvas_size)
        rx = int(params[2] * self.canvas_size / 2)
        ry = int(params[3] * self.canvas_size / 2)
        
        bbox = [cx - rx, cy - ry, cx + rx, cy + ry]
        
        # Couleur de remplissage
        fill_color = (int(params[8] * 255), int(params[9] * 255), 
                     int(params[10] * 255), int(params[12] * 255))
        # Couleur de contour
        outline_color = (int(params[5] * 255), int(params[6] * 255), 
                        int(params[7] * 255), int(params[11] * 255))
        
        draw.ellipse(bbox, fill=fill_color, outline=outline_color, width=2)
    
    def _draw_polygon(self, draw, params):
        """Dessine un polygone (pour les formes angulaires comme les cheveux)"""
        # params: [x1, y1, x2, y2, x3, y3, x4, y4, r, g, b, alpha]
        points = []
        for i in range(0, 8, 2):
            x = int(params[i] * self.canvas_size)
            y = int(params[i+1] * self.canvas_size)
            points.append((x, y))
        
        color = (int(params[8] * 255), int(params[9] * 255), 
                int(params[10] * 255), int(params[11] * 255))
        
        draw.polygon(points, fill=color)
    
    def _apply_gradient(self, img, params):
        """Applique un gradient pour les effets d'ombre et de lumière"""
        # params: [x, y, radius, r1, g1, b1, r2, g2, b2, opacity]
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        cx = int(params[0] * self.canvas_size)
        cy = int(params[1] * self.canvas_size)
        radius = int(params[2] * self.canvas_size / 2)
        
        # Créer un gradient radial
        for r in range(radius, 0, -2):
            alpha = int((1 - r/radius) * params[9] * 255)
            t = r / radius
            color = (
                int((params[3] * t + params[6] * (1-t)) * 255),
                int((params[4] * t + params[7] * (1-t)) * 255),
                int((params[5] * t + params[8] * (1-t)) * 255),
                alpha
            )
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=color)
        
        # Composer avec l'image originale
        img = Image.alpha_composite(img.convert('RGBA'), overlay)
        return img.convert('RGB')
    
    def _apply_texture(self, img, params):
        """Applique une texture/pattern (pour les vêtements, arrière-plans)"""
        # Créer un pattern simple
        pattern = Image.new('RGBA', (20, 20), (0, 0, 0, 0))
        draw = ImageDraw.Draw(pattern)
        
        # Dessiner un motif
        color = (int(params[0] * 255), int(params[1] * 255), 
                int(params[2] * 255), int(params[3] * 255))
        for i in range(0, 20, 4):
            draw.line([(0, i), (20, i)], fill=color, width=1)
            draw.line([(i, 0), (i, 20)], fill=color, width=1)
        
        # Répéter le pattern
        texture = Image.new('RGBA', img.size, (0, 0, 0, 0))
        for x in range(0, img.width, 20):
            for y in range(0, img.height, 20):
                texture.paste(pattern, (x, y))
        
        # Appliquer avec opacité
        img = Image.alpha_composite(img.convert('RGBA'), texture)
        return img.convert('RGB')

class ImprovedStrokePredictor(nn.Module):
    """Réseau amélioré avec attention et compréhension de la structure"""
    
    def __init__(self, hidden_dim=1024):
        super(ImprovedStrokePredictor, self).__init__()
        
        # Utiliser un ResNet pré-entraîné comme encodeur
        resnet = models.resnet34(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        
        # Adapter pour 6 canaux d'entrée
        self.input_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Mécanisme d'attention
        self.attention = nn.MultiheadAttention(512, num_heads=8, batch_first=True)
        
        # Encodeur de contexte global
        self.context_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # LSTM pour la séquence de dessin
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=3, 
                           batch_first=True, dropout=0.3)
        
        # Décodeurs pour différents types de primitives
        self.primitive_type_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 5),  # 5 types de primitives
            nn.Softmax(dim=-1)
        )
        
        # Paramètres pour chaque primitive
        self.bezier_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 16),  # 16 paramètres pour Bézier
            nn.Sigmoid()
        )
        
        self.ellipse_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 13),  # 13 paramètres pour ellipse
            nn.Sigmoid()
        )
        
        self.polygon_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 12),  # 12 paramètres pour polygone
            nn.Sigmoid()
        )
        
        self.gradient_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 10),  # 10 paramètres pour gradient
            nn.Sigmoid()
        )
        
        self.texture_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4),  # 4 paramètres pour texture
            nn.Sigmoid()
        )
        
        # Tête pour décider quand arrêter
        self.stop_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, canvas, target, hidden=None):
        # Concaténer canvas et cible
        x = torch.cat([canvas, target], dim=1)
        
        # Extraction de features
        x = self.input_conv(x)
        features = self.feature_extractor(x)
        
        # Contexte global
        context = self.context_encoder(features)
        context = context.unsqueeze(1)
        
        # LSTM
        lstm_out, hidden = self.lstm(context, hidden)
        lstm_out = lstm_out[:, -1, :]
        
        # Prédire le type de primitive
        primitive_type = self.primitive_type_head(lstm_out)
        
        # Prédire les paramètres pour chaque primitive
        bezier_params = self.bezier_decoder(lstm_out)
        ellipse_params = self.ellipse_decoder(lstm_out)
        polygon_params = self.polygon_decoder(lstm_out)
        gradient_params = self.gradient_decoder(lstm_out)
        texture_params = self.texture_decoder(lstm_out)
        
        # Probabilité d'arrêt
        stop_prob = self.stop_head(lstm_out)
        
        return {
            'primitive_type': primitive_type,
            'bezier': bezier_params,
            'ellipse': ellipse_params,
            'polygon': polygon_params,
            'gradient': gradient_params,
            'texture': texture_params,
            'stop_prob': stop_prob,
            'hidden': hidden
        }

class ImprovedDrawingAgent:
    """Agent amélioré pour dessiner des images d'animé complexes"""
    
    def __init__(self, device='cuda', lr=1e-4):
        self.device = device
        self.env = AdvancedDrawingEnvironment()
        self.model = ImprovedStrokePredictor().to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
    def draw_image(self, target_image, max_strokes=200, training=False):
        """Dessine une image avec des primitives variées"""
        canvas = self.env.reset()
        strokes = []
        hidden = None
        
        canvas_tensor = torch.FloatTensor(canvas).permute(2, 0, 1).unsqueeze(0).to(self.device)
        target_tensor = target_image.unsqueeze(0).to(self.device) if target_image.dim() == 3 else target_image
        
        for step in range(max_strokes):
            # Prédiction
            with torch.set_grad_enabled(training):
                outputs = self.model(canvas_tensor, target_tensor, hidden)
            
            # Vérifier si on doit arrêter
            if outputs['stop_prob'].item() > 0.95 and step > 30:
                break
            
            # Sélectionner le type de primitive
            primitive_type = torch.argmax(outputs['primitive_type']).item()
            
            # Récupérer les paramètres correspondants
            if primitive_type == 0:
                params = outputs['bezier'][0].cpu().detach().numpy()
            elif primitive_type == 1:
                params = outputs['ellipse'][0].cpu().detach().numpy()
            elif primitive_type == 2:
                params = outputs['polygon'][0].cpu().detach().numpy()
            elif primitive_type == 3:
                params = outputs['gradient'][0].cpu().detach().numpy()
            else:
                params = outputs['texture'][0].cpu().detach().numpy()
            
            # Ajouter du bruit pour l'exploration
            if training and random.random() < 0.15:
                params += np.random.normal(0, 0.03, params.shape)
                params = np.clip(params, 0, 1)
            
            # Appliquer la primitive
            canvas = self.env.apply_primitive(primitive_type, params)
            strokes.append((primitive_type, params))
            
            # Mettre à jour le canvas
            canvas_tensor = torch.FloatTensor(canvas).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            # Gérer le hidden state
            hidden = outputs['hidden']
            if training and hidden is not None:
                hidden = (hidden[0].detach(), hidden[1].detach())
        
        return canvas, strokes
    
    def compute_perceptual_loss(self, drawn, target):
        """Calcule une loss perceptuelle avec VGG"""
        # Utiliser VGG pour la loss perceptuelle
        vgg = models.vgg16(pretrained=True).features[:16].to(self.device).eval()
        
        with torch.no_grad():
            drawn_features = vgg(drawn)
            target_features = vgg(target)
        
        return F.mse_loss(drawn_features, target_features)
    
    def train_step(self, target_image):
        """Entraînement avec supervision directe sur les primitives"""
        self.model.train()
        self.optimizer.zero_grad()
        
        canvas = self.env.reset()
        canvas_tensor = torch.FloatTensor(canvas).permute(2, 0, 1).unsqueeze(0).to(self.device)
        target_tensor = target_image.unsqueeze(0).to(self.device) if target_image.dim() == 3 else target_image
        
        total_loss = 0
        hidden = None
        num_strokes = 50
        
        for step in range(num_strokes):
            outputs = self.model(canvas_tensor, target_tensor, hidden)
            
            # Loss sur la distribution des primitives (encourager la diversité)
            entropy_loss = -torch.sum(outputs['primitive_type'] * 
                                     torch.log(outputs['primitive_type'] + 1e-8))
            
            # Loss de régularisation sur les paramètres
            param_loss = 0
            for key in ['bezier', 'ellipse', 'polygon', 'gradient', 'texture']:
                params = outputs[key]
                # Encourager des valeurs modérées
                param_loss += torch.mean((params - 0.5) ** 2) * 0.01
            
            # Loss sur la probabilité d'arrêt
            stop_target = torch.ones_like(outputs['stop_prob']) if step < 20 else torch.zeros_like(outputs['stop_prob'])
            stop_loss = F.binary_cross_entropy(outputs['stop_prob'], stop_target) * 0.1
            
            total_loss = total_loss + entropy_loss * 0.01 + param_loss + stop_loss
            
            # Appliquer la primitive (sans gradient)
            with torch.no_grad():
                primitive_type = torch.argmax(outputs['primitive_type']).item()
                
                if primitive_type == 0:
                    params = outputs['bezier'][0].cpu().numpy()
                elif primitive_type == 1:
                    params = outputs['ellipse'][0].cpu().numpy()
                elif primitive_type == 2:
                    params = outputs['polygon'][0].cpu().numpy()
                elif primitive_type == 3:
                    params = outputs['gradient'][0].cpu().numpy()
                else:
                    params = outputs['texture'][0].cpu().numpy()
                
                canvas = self.env.apply_primitive(primitive_type, params)
                canvas_tensor = torch.FloatTensor(canvas).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            hidden = outputs['hidden']
        
        # Loss de reconstruction finale
        with torch.no_grad():
            drawn_tensor = torch.FloatTensor(canvas).permute(2, 0, 1).unsqueeze(0).to(self.device)
            reconstruction_loss = F.mse_loss(drawn_tensor, target_tensor).item()
            
            # Loss perceptuelle
            perceptual_loss = self.compute_perceptual_loss(drawn_tensor, target_tensor).item()
        
        # Ajouter les losses de reconstruction
        total_loss = total_loss + reconstruction_loss + perceptual_loss * 0.1
        
        # Backpropagation
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item(), canvas

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

def train_drawing_ai(dataset_path, epochs=100, batch_size=4):
    """Fonction principale d'entraînement"""
    
    # Vérifier GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation de: {device}")
    
    # Créer le dataset
    dataset = AnimeDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    # Créer l'agent
    agent = ImprovedDrawingAgent(device=device)
    
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
        
        # Scheduler
        agent.scheduler.step()
        
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

def main():
    # Créer les dossiers nécessaires
    os.makedirs('examples', exist_ok=True)
    
    # Chemin vers votre dataset
    DATASET_PATH = "/kaggle/input/anima-s-dataset/test/"
    
    # Entraîner l'IA
    print("Démarrage de l'entraînement de l'IA de dessin avancée...")
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

if __name__ == "__main__":
    main()