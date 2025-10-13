import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import Callback
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2  # OpenCV est nécessaire pour le filtrage
from shutil import move, rmtree

# ==============================================================================
# PARTIE 1 : FONCTION DE PRÉ-TRAITEMENT DU DATASET
# ==============================================================================

def filter_dataset(source_folder, keep_folder, discard_folder, min_diff=1.5, max_diff=35.0):
    """
    Filtre un dataset de frames pour ne garder que celles avec un mouvement modéré.
    Cette fonction déplace les fichiers, elle ne les copie pas.

    Args:
        source_folder (str): Dossier contenant les frames originales.
        keep_folder (str): Dossier où stocker les frames utiles pour l'entraînement.
        discard_folder (str): Dossier où stocker les frames rejetées (cuts/statiques).
        min_diff (float): Seuil de différence minimal (en %). En dessous, l'image est jugée trop statique.
        max_diff (float): Seuil de différence maximal (en %). Au-dessus, c'est probablement un cut.
    """
    print("--- DÉBUT DU FILTRAGE DU DATASET ---")
    if os.path.exists(keep_folder): rmtree(keep_folder)
    if os.path.exists(discard_folder): rmtree(discard_folder)
    os.makedirs(keep_folder, exist_ok=True)
    os.makedirs(discard_folder, exist_ok=True)

    files = sorted([f for f in os.listdir(source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not files:
        print(f"ERREUR: Aucun fichier image trouvé dans '{source_folder}'.")
        return

    prev_frame_gray = None
    kept_files_count = 0

    for i, filename in enumerate(files):
        filepath = os.path.join(source_folder, filename)
        frame = cv2.imread(filepath)
        if frame is None:
            print(f"Attention: Impossible de lire le fichier {filename}, il sera ignoré.")
            continue
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # La toute première frame est toujours gardée comme point de départ
        if i == 0:
            prev_frame_gray = frame_gray
            move(filepath, os.path.join(keep_folder, filename))
            kept_files_count += 1
            continue

        # Calcul de la différence avec la frame précédente
        diff = cv2.absdiff(prev_frame_gray, frame_gray)
        non_zero_count = np.count_nonzero(diff)
        diff_percent = (non_zero_count / diff.size) * 100

        # Décision de garder ou de jeter la frame
        if min_diff < diff_percent < max_diff:
            print(f"CONSERVER : {filename} (Différence: {diff_percent:.2f}%)")
            move(filepath, os.path.join(keep_folder, filename))
            kept_files_count += 1
        else:
            reason = "STATIQUE" if diff_percent <= min_diff else "CUT"
            print(f"REJETER   : {filename} (Différence: {diff_percent:.2f}%) - Raison: {reason}")
            move(filepath, os.path.join(discard_folder, filename))
        
        prev_frame_gray = frame_gray
    
    print(f"--- FILTRAGE TERMINÉ ---")
    print(f"Total des images conservées : {kept_files_count} / {len(files)}")
    print(f"Les images pour l'entraînement sont dans le dossier : '{keep_folder}'")


# ==============================================================================
# PARTIE 2 : MODÈLE U-NET ET ENTRAÎNEMENT
# ==============================================================================

def unet_model(input_size=(128, 128, 3)):
    """Création d'un modèle U-Net standard."""
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

def load_data(data_path, img_size=(128, 128)):
    """Charge les images et crée les paires (frame_N, frame_N+1)."""
    X, y = [], []
    image_files = sorted([f for f in os.listdir(data_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
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

class SaveEpochPrediction(Callback):
    """Callback pour sauvegarder une prédiction à la fin de chaque époque."""
    def __init__(self, test_image, save_path):
        super(Callback, self).__init__()
        self.test_image = np.expand_dims(test_image, axis=0)
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        predicted_frame = self.model.predict(self.test_image, verbose=0)[0]
        predicted_frame = (predicted_frame * 255).astype(np.uint8)
        img = Image.fromarray(predicted_frame)
        img.save(os.path.join(self.save_path, f'epoch_{epoch+1:03d}.png'))

# ==============================================================================
# PARTIE 3 : EXÉCUTION
# ==============================================================================

if __name__ == '__main__':
    # --- CONFIGURATION ---
    IMAGE_HEIGHT = 128
    IMAGE_WIDTH = 128
    
    # 1. Dossier contenant TOUTES vos frames originales
    SOURCE_DATA_PATH = 'anime_frames_original'
    
    # 2. Dossier qui contiendra les frames "propres" après filtrage
    CLEANED_DATA_PATH = 'anime_frames_cleaned'
    
    # 3. Dossier pour les frames rejetées (pour vérification)
    DISCARDED_DATA_PATH = 'anime_frames_discarded'
    
    # 4. Dossier pour sauvegarder les résultats du modèle
    RESULTS_PATH = 'results'

    # --- ÉTAPE 1: FILTRAGE (à exécuter une seule fois) ---
    # Décommentez la ligne ci-dessous pour lancer le filtrage.
    # Une fois terminé, vous pouvez la re-commenter.
    # --------------------------------------------------------------------------
    # filter_dataset(SOURCE_DATA_PATH, CLEANED_DATA_PATH, DISCARDED_DATA_PATH)
    # --------------------------------------------------------------------------

    # --- ÉTAPE 2: ENTRAÎNEMENT DU MODÈLE ---
    # Assurez-vous que le filtrage a été fait et que le dossier CLEANED_DATA_PATH existe.
    if not os.path.exists(CLEANED_DATA_PATH) or not os.listdir(CLEANED_DATA_PATH):
        print(f"\nERREUR: Le dossier '{CLEANED_DATA_PATH}' est vide ou n'existe pas.")
        print("Veuillez d'abord exécuter l'étape de filtrage en décommentant la ligne `filter_dataset(...)`.")
        exit()

    print("\n--- DÉBUT DE L'ENTRAÎNEMENT DU MODÈLE ---")
    
    # Création des dossiers de résultats
    EPOCH_PREVIEWS_PATH = os.path.join(RESULTS_PATH, 'epoch_previews')
    os.makedirs(EPOCH_PREVIEWS_PATH, exist_ok=True)
        
    # Chargement des données NETTOYÉES
    try:
        X_train, y_train = load_data(CLEANED_DATA_PATH, img_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
    except ValueError as e:
        print(f"ERREUR : {e}")
        exit()
        
    # Création du modèle
    model = unet_model(input_size=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mae']) # Utilisation de MAE (L1), souvent meilleure pour les images
    model.summary()
    
    # Création du Callback pour sauvegarder la progression
    save_callback = SaveEpochPrediction(test_image=X_train[0], save_path=EPOCH_PREVIEWS_PATH)

    # Entraînement
    history = model.fit(
        X_train, 
        y_train, 
        epochs=150,       # Augmentez si nécessaire
        batch_size=8,       # Ajustez selon la VRAM de votre GPU
        validation_split=0.1,
        callbacks=[save_callback]
    )
    
    # Sauvegarde du modèle final
    FINAL_MODEL_PATH = os.path.join(RESULTS_PATH, 'unet_anime_predictor_v2.h5')
    print(f"\nEntraînement terminé. Sauvegarde du modèle final à : {FINAL_MODEL_PATH}")
    model.save(FINAL_MODEL_PATH)

    # Prédiction et visualisation finale
    print("\nCréation de l'image de comparaison finale...")
    input_frame = np.expand_dims(X_train[0], axis=0)
    predicted_frame = model.predict(input_frame)[0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(X_train[0]); axes[0].set_title("Frame d'Entrée (N)"); axes[0].axis('off')
    axes[1].imshow(y_train[0]); axes[1].set_title("Frame Réelle Suivante (N+1)"); axes[1].axis('off')
    axes[2].imshow(predicted_frame); axes[2].set_title("Frame Prédite Finale"); axes[2].axis('off')
    
    final_comparison_path = os.path.join(RESULTS_PATH, 'final_prediction_comparison.png')
    plt.savefig(final_comparison_path)
    print(f"Image de comparaison finale enregistrée à : {final_comparison_path}")
    plt.show()