import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# --- 1. Définition de l'Architecture du U-Net ---

def unet_model(input_size=(256, 256, 3)):
    """
    Création d'un modèle U-Net standard.
    """
    inputs = layers.Input(input_size)

    # --- Encodeur (Partie de contraction) ---
    c1 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = layers.Dropout(0.1)(c1)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = layers.Dropout(0.1)(c2)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # --- Bottleneck ---
    c5 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c5 = layers.Dropout(0.2)(c5)
    c5 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # --- Décodeur (Partie d'expansion) avec les skip connections ---
    u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c2]) # Skip connection
    c6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = layers.Dropout(0.2)(c6)
    c6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c1]) # Skip connection
    c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = layers.Dropout(0.1)(c7)
    c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    # La couche de sortie doit avoir le même nombre de canaux que l'image de sortie (3 pour RGB)
    # L'activation 'sigmoid' force les pixels à être entre 0 et 1.
    outputs = layers.Conv2D(3, (1, 1), activation='sigmoid')(c7)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# --- 2. Préparation des Données ---

def load_data(data_path, img_size=(256, 256)):
    """
    Charge les images et crée les paires (frame_N, frame_N+1).
    Les images doivent être nommées de manière séquentielle (ex: frame_001.png, frame_002.png).
    """
    X, y = [], []
    image_files = sorted(os.listdir(data_path))

    for i in range(len(image_files) - 1):
        # Frame actuelle (entrée X)
        img_path_current = os.path.join(data_path, image_files[i])
        img_current = Image.open(img_path_current).resize(img_size)
        X.append(np.array(img_current))

        # Frame suivante (sortie y)
        img_path_next = os.path.join(data_path, image_files[i+1])
        img_next = Image.open(img_path_next).resize(img_size)
        y.append(np.array(img_next))

    # Normalisation des pixels entre 0 et 1
    X = np.array(X) / 255.0
    y = np.array(y) / 255.0

    return X, y

# --- 3. Entraînement et Prédiction ---

if __name__ == '__main__':
    # --- Configuration ---
    IMAGE_HEIGHT = 128
    IMAGE_WIDTH = 128
    # Créez un dossier 'data' et placez-y vos images séquentielles
    DATA_PATH = 'data'
    
    # Création de données factices pour l'exemple si le dossier n'existe pas
    if not os.path.exists(DATA_PATH):
        print("Création de données de démonstration...")
        os.makedirs(DATA_PATH)
        for i in range(10):
            # Crée une image avec un carré qui se déplace
            img = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
            x_pos = i * 10
            img[50:70, x_pos:x_pos+20, :] = 255 # Carré blanc
            Image.fromarray(img).save(os.path.join(DATA_PATH, f'frame_{i:03d}.png'))
            
    # --- Chargement des données ---
    X_train, y_train = load_data(DATA_PATH, img_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
    print(f"Données chargées : {len(X_train)} paires d'images.")

    # --- Création et Compilation du Modèle ---
    model = unet_model(input_size=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    # 'Adam' est un bon optimiseur par défaut.
    # 'mean_squared_error' est une excellente fonction de perte pour la régression d'images.
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    model.summary()

    # --- Entraînement ---
    print("\nDébut de l'entraînement...")
    # 'epochs' est le nombre de fois que le modèle verra l'ensemble des données.
    # 'batch_size' est le nombre d'images traitées à la fois.
    history = model.fit(X_train, y_train, epochs=50, batch_size=4, validation_split=0.1)

    # --- Prédiction et Visualisation ---
    print("\nPrédiction sur la première image de l'ensemble de test...")
    # On utilise la première image de l'ensemble pour prédire la seconde
    input_frame = np.expand_dims(X_train[0], axis=0)
    predicted_frame = model.predict(input_frame)[0]

    # --- Affichage des résultats ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Image d'entrée
    axes[0].imshow(X_train[0])
    axes[0].set_title('Frame d\'Entrée (N)')
    axes[0].axis('off')

    # Image réelle suivante
    axes[1].imshow(y_train[0])
    axes[1].set_title('Frame Réelle Suivante (N+1)')
    axes[1].axis('off')

    # Image prédite par le modèle
    axes[2].imshow(predicted_frame)
    axes[2].set_title('Frame Prédite par le U-Net')
    axes[2].axis('off')

    plt.show()