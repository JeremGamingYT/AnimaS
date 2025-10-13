from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# Nous utilisons un KNN pour améliorer la qualité des prédictions.
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split


def load_frames(data_dir: str) -> List[np.ndarray]:
    """Charge et retourne toutes les images PNG du dossier donné.

    Les images sont triées par ordre alphabétique (supposé correspondre
    à l'ordre chronologique). Chaque image est convertie en niveaux de gris
    et normalisée dans [0,1].

    Args:
        data_dir: chemin du dossier contenant les images.

    Returns:
        Liste de tableaux numpy de shape (H, W) en float32.
    """
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(".png")]
    files.sort()
    frames = []
    for fname in files:
        path = os.path.join(data_dir, fname)
        img = Image.open(path).convert("L")  # niveaux de gris
        arr = np.array(img, dtype=np.float32) / 255.0
        frames.append(arr)
    return frames


def prepare_dataset(frames: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Prépare les paires (frame_t, frame_{t+1}).

    Args:
        frames: liste d'images dans l'ordre chronologique.

    Returns:
        Tuple (X, Y) où X correspond aux frames d'entrée et Y aux
        frames cibles (images suivantes). Les dimensions sont
        (n_pairs, H*W).
    """
    if len(frames) < 2:
        raise ValueError("Au moins deux images sont nécessaires pour créer des paires.")
    images = np.stack(frames)
    # Créer les paires successives
    X = images[:-1]
    Y = images[1:]
    n_samples, H, W = X.shape
    # Aplatir
    return X.reshape(n_samples, H * W), Y.reshape(n_samples, H * W)


def train_pca_regression(
    X: np.ndarray, Y: np.ndarray, n_components: int, test_size: float = 0.2, random_state: int = 0
) -> Tuple[PCA, LinearRegression, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Entraîne une PCA et un modèle KNN pour prédire Y à partir de X.

    Args:
        X: frames d'entrée aplaties (n_samples, n_features).
        Y: frames de sortie aplaties (n_samples, n_features).
        n_components: nombre de composantes principales à conserver.
        test_size: fraction du jeu réservée au test.
        random_state: graine pour la séparation des données.

    Returns:
        Tuple (pca, reg, X_test_pca, Y_test_pca, X_test, Y_test) contenant la PCA
        ajustée, le KNN entraîné, les représentations PCA du jeu de test
        et les images aplaties correspondantes pour évaluation.
    """
    # Limiter le nombre de composantes au maximum autorisé
    max_components = min(n_components, min(X.shape[0] + Y.shape[0] - 2, X.shape[1]))
    pca = PCA(n_components=max_components)
    # Ajuster la PCA sur l'ensemble des frames (entrées et sorties)
    pca.fit(np.concatenate([X, Y], axis=0))
    # Transformer
    X_pca = pca.transform(X)
    Y_pca = pca.transform(Y)
    # Séparation entraînement/test
    X_train_pca, X_test_pca, Y_train_pca, Y_test_pca, X_train, X_test, Y_train, Y_test = train_test_split(
        X_pca, Y_pca, X, Y, test_size=test_size, random_state=random_state
    )
    # Modèle KNN : pour chaque représentation PCA, on cherche les k plus
    # proches voisins dans l’espace de représentation et on moyenne leurs
    # frames suivantes. Les poids basés sur la distance améliorent les
    # prédictions en donnant plus d’importance aux voisins les plus proches.
    knn_reg = KNeighborsRegressor(n_neighbors=3, weights="distance")
    knn_reg.fit(X_train_pca, Y_train_pca)
    return pca, knn_reg, X_test_pca, Y_test_pca, X_test, Y_test


def reconstruct_and_plot(
    pca: PCA,
    reg: LinearRegression,
    X_test_pca: np.ndarray,
    Y_test_pca: np.ndarray,
    num_examples: int = 5,
    save_path: str | None = None,
    original_shape: Tuple[int, int] = (256, 256),
) -> None:
    """Reconstruit et affiche quelques exemples de prédictions.

    Args:
        pca: modèle PCA entraîné.
        reg: modèle de régression linéaire entraîné.
        X_test_pca: représentations PCA des frames d'entrée de test.
        Y_test_pca: représentations PCA des frames réelles de test.
        num_examples: nombre d'exemples à afficher.
        save_path: chemin pour sauvegarder la figure. Si None, la figure n'est
            pas sauvegardée.
        original_shape: tuple (H, W) de la taille des images originales.
    """
    n_examples = min(num_examples, len(X_test_pca))
    H, W = original_shape
    fig, axes = plt.subplots(n_examples, 3, figsize=(8, 2.5 * n_examples))
    if n_examples == 1:
        axes = np.expand_dims(axes, axis=0)
    preds_pca = reg.predict(X_test_pca[:n_examples])
    for i in range(n_examples):
        inp_img = pca.inverse_transform(X_test_pca[i]).reshape(H, W)
        true_img = pca.inverse_transform(Y_test_pca[i]).reshape(H, W)
        pred_img = pca.inverse_transform(preds_pca[i]).reshape(H, W)
        axes[i, 0].imshow(inp_img, cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title("Frame 1")
        axes[i, 1].imshow(true_img, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title("Frame 2 (réelle)")
        axes[i, 2].imshow(np.clip(pred_img, 0, 1), cmap="gray", vmin=0, vmax=1)
        axes[i, 2].set_title("Frame 2 (prédite)")
        for j in range(3):
            axes[i, j].axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prédiction de la frame suivante sur un jeu d'images animées.")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Chemin du dossier contenant les images (frames).",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=50,
        help="Nombre de composantes principales à conserver pour la PCA.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction du jeu utilisée pour le test (entre 0 et 1).",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=5,
        help="Nombre d'exemples à afficher dans la figure finale.",
    )
    args = parser.parse_args()
    # Chargement des images
    print(f"Chargement des images depuis {args.data_dir}…")
    frames = load_frames(args.data_dir)
    H, W = frames[0].shape
    print(f"{len(frames)} images chargées. Taille des images : {H}x{W}.")
    # Préparation des paires
    X, Y = prepare_dataset(frames)
    print(f"Nombre de paires (frame_t, frame_t+1) : {len(X)}")
    # Entraînement PCA + régression
    pca, reg, X_test_pca, Y_test_pca, X_test, Y_test = train_pca_regression(
        X, Y, n_components=args.n_components, test_size=args.test_size
    )
    # Calcul de l'erreur MSE en espace PCA
    mse_pca = np.mean((reg.predict(X_test_pca) - Y_test_pca) ** 2)
    print(f"Erreur quadratique moyenne dans l'espace PCA : {mse_pca:.4f}")
    # Affichage et sauvegarde
    output_path = "predictions_anime.png"
    print(f"Sauvegarde de la figure des prédictions dans {output_path}…")
    reconstruct_and_plot(
        pca,
        reg,
        X_test_pca,
        Y_test_pca,
        num_examples=args.examples,
        save_path=output_path,
        original_shape=(H, W),
    )
    print("Terminé.")


if __name__ == "__main__":
    main()