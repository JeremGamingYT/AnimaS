import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.callbacks import Callback
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# --- 1. Définition de l'Architecture du U-Net (inchangée) ---

def unet_model(input_size=(128, 128, 3)):
    """
    Création d'un modèle U-Net standard.
    """
    inputs = layers.Input(input_size)
    # Encodeur
    c1 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = layers.Dropout(0.1)(c1)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = layers.Dropout(0.1)(c2)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    # Bottleneck
    c5 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c5 = layers.Dropout(0.2)(c5)
    c5 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    # Décodeur
    u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = layers.Dropout(0.2)(c6)
    c6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = layers.Dropout(0.1)(c7)
    c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    outputs = layers.Conv2D(3, (1, 1), activation='sigmoid')(c7)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# --- 2. Préparation des Données (inchangée) ---

def load_data(data_path, img_size=(128, 128)):
    """
    Charge les images et crée les paires (frame_N, frame_N+1).
    """
    X, y = [], []
    image_files = sorted([f for f in os.listdir(data_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if len(image_files) < 2:
        raise ValueError(f"Le dossier '{data_path}' doit contenir au moins 2 images pour créer une paire.")

    print(f"Chargement de {len(image_files) - 1} paires d'images depuis '{data_path}'...")
    for i in range(len(image_files) - 1):
        img_path_current = os.path.join(data_path, image_files[i])
        img_current = Image.open(img_path_current).convert('RGB').resize(img_size)
        X.append(np.array(img_current))

        img_path_next = os.path.join(data_path, image_files[i+1])
        img_next = Image.open(img_path_next).convert('RGB').resize(img_size)
        y.append(np.array(img_next))

    X = np.array(X) / 255.0
    y = np.array(y) / 255.0
    return X, y

# --- 3. NOUVEAU : Callback pour sauvegarder les prédictions à chaque époque ---

class SaveEpochPrediction(Callback):
    def __init__(self, test_image, save_path):
        super(Callback, self).__init__()
        self.test_image = np.expand_dims(test_image, axis=0)
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        predicted_frame = self.model.predict(self.test_image)[0]
        # Dénormaliser l'image pour la sauvegarde (0-255)
        predicted_frame = (predicted_frame * 255).astype(np.uint8)
        img = Image.fromarray(predicted_frame)
        img.save(os.path.join(self.save_path, f'epoch_{epoch+1:03d}.png'))
        print(f"\nImage de prédiction pour l'époque {epoch+1} enregistrée.")


# --- 4. Script Principal ---

if __name__ == '__main__':
    # --- Configuration ---
    IMAGE_HEIGHT = 128
    IMAGE_WIDTH = 128
    
    # !! MODIFIEZ CECI pour pointer vers votre dossier de frames d'animé !!
    DATA_PATH = 'anime_frames' 
    
    # Création des dossiers pour les résultats
    RESULTS_PATH = 'results'
    EPOCH_PREVIEWS_PATH = os.path.join(RESULTS_PATH, 'epoch_previews')
    os.makedirs(EPOCH_PREVIEWS_PATH, exist_ok=True)
    
    # Vérification du dossier de données
    if not os.path.exists(DATA_PATH):
        print(f"ERREUR : Le dossier '{DATA_PATH}' n'a pas été trouvé.")
        print("Veuillez créer ce dossier et y placer vos frames d'animé.")
        exit() # Arrête le script si le dossier n'existe pas

    # --- Chargement des données ---
    try:
        X_train, y_train = load_data(DATA_PATH, img_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
    except ValueError as e:
        print(f"ERREUR : {e}")
        exit()
        
    print(f"Données chargées. Forme de X_train : {X_train.shape}, Forme de y_train : {y_train.shape}")

    # --- Création et Compilation du Modèle ---
    model = unet_model(input_size=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    model.summary()
    
    # --- Création du Callback ---
    # Nous utiliserons la première image du set pour visualiser la progression
    save_callback = SaveEpochPrediction(test_image=X_train[0], save_path=EPOCH_PREVIEWS_PATH)

    # --- Entraînement ---
    print("\nDébut de l'entraînement...")
    history = model.fit(
        X_train, 
        y_train, 
        epochs=100, # Augmentez si nécessaire pour votre dataset
        batch_size=8,  # Ajustez selon la mémoire de votre GPU
        validation_split=0.1,
        callbacks=[save_callback] # Ajout du callback ici !
    )
    
    # --- Sauvegarde du Modèle Final ---
    FINAL_MODEL_PATH = os.path.join(RESULTS_PATH, 'unet_anime_predictor.h5')
    print(f"\nEntraînement terminé. Sauvegarde du modèle final à : {FINAL_MODEL_PATH}")
    model.save(FINAL_MODEL_PATH)

    # --- Prédiction et Visualisation Finale ---
    print("\nCréation de l'image de comparaison finale...")
    input_frame = np.expand_dims(X_train[0], axis=0)
    predicted_frame = model.predict(input_frame)[0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(X_train[0])
    axes[0].set_title('Frame d\'Entrée (N)')
    axes[0].axis('off')

    axes[1].imshow(y_train[0])
    axes[1].set_title('Frame Réelle Suivante (N+1)')
    axes[1].axis('off')

    axes[2].imshow(predicted_frame)
    axes[2].set_title('Frame Prédite Finale')
    axes[2].axis('off')

    # Sauvegarde de la figure
    final_comparison_path = os.path.join(RESULTS_PATH, 'final_prediction_comparison.png')
    plt.savefig(final_comparison_path)
    print(f"Image de comparaison finale enregistrée à : {final_comparison_path}")
    plt.show()