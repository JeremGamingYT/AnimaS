import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import torchvision.transforms as transforms

# Définition de l'architecture U-Net
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # Encodeur
        self.enc1 = CBR(in_channels, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Pont
        self.bridge = CBR(512, 1024)

        # Décodeur
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = CBR(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = CBR(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = CBR(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = CBR(128, 64)

        # Couche de sortie
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encodeur
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Pont
        bridge = self.bridge(self.pool(e4))

        # Décodeur
        d4 = self.upconv4(bridge)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)

# Création d'un jeu de données factice pour l'exemple
class FrameDataset(Dataset):
    def __init__(self, frame_dir, transform=None):
        self.frame_dir = frame_dir
        self.transform = transform
        self.frames = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.frames) - 1

    def __getitem__(self, idx):
        frame1_path = os.path.join(self.frame_dir, self.frames[idx])
        frame2_path = os.path.join(self.frame_dir, self.frames[idx+1])

        frame1 = Image.open(frame1_path).convert("RGB")
        frame2 = Image.open(frame2_path).convert("RGB")

        if self.transform:
            frame1 = self.transform(frame1)
            frame2 = self.transform(frame2)

        return frame1, frame2

# Fonction pour générer des images factices
def generate_dummy_frames(num_frames, size=(128, 128)):
    if not os.path.exists('dummy_frames'):
        os.makedirs('dummy_frames')
    for i in range(num_frames):
        # Crée une image avec un carré qui se déplace
        img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        x_pos = int((i / num_frames) * (size[1] - 20))
        img[50:70, x_pos:x_pos+20] = [255, 0, 0] # Carré rouge
        img = Image.fromarray(img)
        img.save(f'dummy_frames/frame_{i:04d}.png')

# Paramètres
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_EPOCHS = 25
IMG_SIZE = 128
NUM_FRAMES = 50

# Générer les images factices
generate_dummy_frames(NUM_FRAMES, size=(IMG_SIZE, IMG_SIZE))

# Transformations des images
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# Création du jeu de données et du chargeur de données
dataset = FrameDataset(frame_dir='/kaggle/input/anima-s-dataset/test', transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialisation du modèle, de la fonction de perte et de l'optimiseur
model = UNet(in_channels=3, out_channels=3).to(DEVICE)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Boucle d'entraînement
for epoch in range(NUM_EPOCHS):
    for batch_idx, (data, targets) in enumerate(loader):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)

        # Forward
        predictions = model(data)
        loss = loss_fn(predictions, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}")

# Exemple de prédiction
model.eval()
with torch.no_grad():
    # Prendre la première frame comme entrée
    input_frame, _ = dataset[0]
    input_frame = input_frame.unsqueeze(0).to(DEVICE)

    # Générer la prédiction
    predicted_frame_tensor = model(input_frame)

    # Sauvegarder l'image prédite
    predicted_frame_tensor = predicted_frame_tensor.squeeze(0).cpu()
    predicted_frame_img = transforms.ToPILImage()(predicted_frame_tensor)
    predicted_frame_img.save("predicted_frame.png")
    print("Prédiction de la frame suivante sauvegardée sous 'predicted_frame.png'")