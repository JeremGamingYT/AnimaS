from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def generate_moving_square_dataset(
    num_samples: int,
    image_size: int = 16,
    square_size: int = 4,
    max_speed: int = 2,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Génère un jeu de données de carrés en mouvement.

    Chaque échantillon est composé de deux images : une frame initiale et
    la frame suivante où le carré a été déplacé d'un certain vecteur.

    Args:
        num_samples: nombre de paires (frame_1, frame_2) à générer.
        image_size: taille des images carrées (hauteur et largeur).
        square_size: taille du carré blanc en mouvement.
        max_speed: déplacement maximal en pixels entre frame_1 et frame_2
            pour chaque dimension (le déplacement est choisi aléatoirement
            entre [-max_speed, max_speed]).
        random_state: graine pour la reproductibilité.

    Returns:
        Tuple `(X, Y)` où `X` est un tableau de shape `(num_samples, image_size, image_size)`
        contenant les frames initiales et `Y` est un tableau de shape
        `(num_samples, image_size, image_size)` contenant les frames suivantes.
    """
    rng = np.random.default_rng(random_state)
    X = np.zeros((num_samples, image_size, image_size), dtype=np.float32)
    Y = np.zeros_like(X)
    for i in range(num_samples):
        # Position initiale du carré
        # On s'assure que le carré est entièrement à l'intérieur de l'image
        x0 = rng.integers(0, image_size - square_size)
        y0 = rng.integers(0, image_size - square_size)
        # Déplacement aléatoire
        dx = rng.integers(-max_speed, max_speed + 1)
        dy = rng.integers(-max_speed, max_speed + 1)
        # Dessiner la première frame
        X[i, y0 : y0 + square_size, x0 : x0 + square_size] = 1.0
        # Calculer la nouvelle position en veillant à rester dans les limites
        x1 = np.clip(x0 + dx, 0, image_size - square_size)
        y1 = np.clip(y0 + dy, 0, image_size - square_size)
        # Dessiner la seconde frame
        Y[i, y1 : y1 + square_size, x1 : x1 + square_size] = 1.0
    return X, Y


def train_model(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    hidden_layer_sizes: tuple[int, ...] = (256, 256),
    max_iter: int = 200,
    random_state: int | None = None,
) -> MLPRegressor:
    """Entraîne un réseau neuronal MLP pour prédire la frame suivante.

    Args:
        X_train: tableau `(n_samples, H, W)` des frames d'entrée.
        Y_train: tableau `(n_samples, H, W)` des frames de sortie.
        hidden_layer_sizes: taille des couches cachées de l'MLP.
        max_iter: nombre maximal d'itérations (époques) pour l'entraînement.
        random_state: graine pour la reproductibilité.

    Returns:
        Un objet `MLPRegressor` entraîné.
    """
    # Aplatir les images pour les passer dans un MLP dense
    n_samples, H, W = X_train.shape
    X_flat = X_train.reshape(n_samples, H * W)
    Y_flat = Y_train.reshape(n_samples, H * W)
    # L'activation ReLU fonctionne bien pour des valeurs dans [0,1]
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        random_state=random_state,
    )
    model.fit(X_flat, Y_flat)
    return model


def evaluate_model(model: MLPRegressor, X_test: np.ndarray, Y_test: np.ndarray) -> float:
    """Évalue le modèle sur un jeu de test et retourne l'erreur quadratique moyenne (MSE).

    Args:
        model: modèle entraîné.
        X_test: frames d'entrée (format `(n_samples, H, W)`).
        Y_test: frames cibles (format `(n_samples, H, W)`).

    Returns:
        Erreur quadratique moyenne entre les sorties prédites et les cibles.
    """
    n_samples, H, W = X_test.shape
    preds = model.predict(X_test.reshape(n_samples, H * W))
    mse = mean_squared_error(Y_test.reshape(n_samples, H * W), preds)
    return mse


def plot_predictions(
    model: MLPRegressor,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    num_examples: int = 5,
    save_path: str | None = None,
) -> None:
    """Affiche quelques prédictions du modèle sous forme d'images.

    On crée une figure matplotlib avec trois colonnes : la première montre la
    frame d'entrée, la deuxième la vraie frame suivante et la troisième la
    prédiction du modèle. Les valeurs sont binarisées pour une meilleure
    visibilité.

    Args:
        model: modèle entraîné.
        X_test: frames d'entrée.
        Y_test: frames cibles.
        num_examples: nombre d'exemples à afficher.
        save_path: chemin où sauvegarder la figure (PNG). Si None, la figure
            n'est pas sauvegardée.
    """
    num_examples = min(num_examples, len(X_test))
    fig, axes = plt.subplots(num_examples, 3, figsize=(6, 2 * num_examples))
    if num_examples == 1:
        axes = np.expand_dims(axes, axis=0)  # assurer une dimension cohérente
    n_samples, H, W = X_test.shape
    preds = model.predict(X_test.reshape(n_samples, H * W))
    preds = preds.reshape(n_samples, H, W)
    # Appliquer un seuillage pour binariser les prédictions (facultatif)
    preds_bin = (preds >= 0.5).astype(float)
    for i in range(num_examples):
        axes[i, 0].imshow(X_test[i], cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title("Frame 1")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(Y_test[i], cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title("Frame 2 (réelle)")
        axes[i, 1].axis("off")
        axes[i, 2].imshow(preds_bin[i], cmap="gray", vmin=0, vmax=1)
        axes[i, 2].set_title("Frame 2 (prédite)")
        axes[i, 2].axis("off")
    plt.tight_layout()
    # Enregistre l'image si un chemin est fourni. Dans des environnements sans
    # interface graphique (comme ce backend), l'appel à ``plt.show()`` ne fera
    # rien. L'enregistrement de la figure permet d'inspecter les résultats
    # après exécution.
    if save_path:
        plt.savefig(save_path)
    # Dans un backend non interactif, plt.show() n'affichera rien mais est
    # maintenu ici pour compatibilité. On ferme ensuite la figure pour libérer
    # la mémoire.
    plt.show()
    plt.close(fig)


def main() -> None:
    """Point d'entrée principal du script.

    Génère un jeu de données de carrés en mouvement, divise en
    entraînement/test, entraîne un modèle MLP, puis évalue et affiche
    quelques prédictions.
    """
    # Paramètres
    num_samples = 800  # nombre total d'exemples générés
    image_size = 16
    square_size = 4
    max_speed = 2
    test_size = 0.2
    random_state = 42

    print("Génération du jeu de données…")
    X, Y = generate_moving_square_dataset(
        num_samples=num_samples,
        image_size=image_size,
        square_size=square_size,
        max_speed=max_speed,
        random_state=random_state,
    )
    # Division en jeu d'entraînement et de test
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )
    print(f"Jeu d'entraînement : {len(X_train)} exemples, jeu de test : {len(X_test)} exemples.")

    print("Entraînement du modèle…")
    model = train_model(
        X_train, Y_train, hidden_layer_sizes=(256, 128), max_iter=200, random_state=random_state
    )
    mse = evaluate_model(model, X_test, Y_test)
    print(f"Erreur quadratique moyenne sur le jeu de test : {mse:.4f}")

    print("Affichage de quelques prédictions…")
    # Sauvegarder la figure des prédictions dans un fichier pour pouvoir
    # l'afficher ultérieurement dans cet environnement.
    pred_fig_path = "predictions.png"
    plot_predictions(model, X_test, Y_test, num_examples=5, save_path=pred_fig_path)
    print(f"Figure enregistrée dans {pred_fig_path}")


if __name__ == "__main__":
    main()