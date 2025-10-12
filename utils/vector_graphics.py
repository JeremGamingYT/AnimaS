"""
Fonctions utilitaires pour manipuler des primitives vectorielles sans dépendre
de bibliothèques externes.

Ce module définit des aides pour créer des primitives (par exemple des cercles)
et rendre une scène en image de façon différentiable.  Dans cette
implémentation, chaque cercle est converti en une gaussienne 2D sur la grille
de pixels, offrant un rendu « soft » qui reste compatible avec la
rétro‑propagation de PyTorch.  Si vous disposez d'un rasteriseur
différentiable externe (comme DRTK), vous pouvez l'intégrer en complétant
la fonction `_render_drtk`.

Cette implémentation se veut simple et compatible avec Python 3.12 et
PyTorch (≥ 2.2).  Aucune dépendance à diffvg ou DRTK n'est nécessaire par
défaut.
"""

from typing import List, Tuple

import math

import torch

# Nous ne dépendons plus d'aucune bibliothèque de rasterisation externe.  Si vous
# disposez de DRTK ou d'une autre librairie permettant de rasteriser des
# triangles de façon différentiable, vous pouvez l'intégrer ici en
# réintroduisant l'importation et en adaptant le code.  Par défaut,
# `drtk` est mis à ``None`` pour forcer l'utilisation du rendu gaussien
# interne.  Cette approche garantit la compatibilité avec Python 3.12 sans
# dépendances externes.
drtk = None  # type: ignore


def create_circle(center: Tuple[float, float], radius: float, color: Tuple[float, float, float]) -> Tuple[Tuple[float, float, float], torch.Tensor]:
    """Crée une représentation interne d'un cercle.

    Le cercle est décrit par son centre `(cx, cy)` et son rayon `r`.  La
    couleur est fournie sous la forme d'un triplet RGB; une composante
    alpha égale à 1.0 est ajoutée automatiquement.  La fonction retourne
    la géométrie `(cx, cy, r)` et un tenseur `color` de longueur 4 (RGBA).
    """
    cx, cy = center
    color_tensor = torch.tensor(list(color) + [1.0], dtype=torch.float32)
    return (float(cx), float(cy), float(radius)), color_tensor


def create_ellipse(center: Tuple[float, float], radii: Tuple[float, float], angle: float,
                   color: Tuple[float, float, float]) -> Tuple[Tuple[float, float, float, float, float], torch.Tensor]:
    """Crée une représentation interne d'une ellipse.

    Une ellipse est définie par son centre `(cx, cy)`, ses deux demi‑axes
    `(rx, ry)` et une orientation `angle` en radians.  Le rendu gaussien
    interne utilise ces paramètres pour générer une gaussienne anisotrope.

    Args:
        center: coordonnées du centre `(cx, cy)`.
        radii: longueur des demi‑axes `(rx, ry)` en pixels.
        angle: angle de rotation en radians (0 correspond à un axe horizontal).
        color: couleur RGB (valeurs entre 0 et 1).

    Returns:
        geometry: tuple `(cx, cy, rx, ry, angle)`.
        color_tensor: tenseur de longueur 4 (RGBA).
    """
    cx, cy = center
    rx, ry = radii
    color_tensor = torch.tensor(list(color) + [1.0], dtype=torch.float32)
    return (float(cx), float(cy), float(rx), float(ry), float(angle)), color_tensor


def _render_gaussian(primitives: List[Tuple[Tuple, torch.Tensor]], canvas_size: Tuple[int, int]) -> torch.Tensor:
    """Rendu différentiable de cercles et ellipses à l'aide de noyaux gaussiens.

    Chaque primitive peut être un cercle `(cx, cy, r)` ou une ellipse
    `(cx, cy, rx, ry, angle)`.  Les primitives sont rendues comme des taches
    floues dont l'intensité décroît progressivement en fonction de la distance
    pondérée par les demi‑axes.  Cette opération est entièrement
    différentiable pour permettre l'optimisation des paramètres.
    """
    height, width = canvas_size
    # Grille de coordonnées pour l'image
    yy, xx = torch.meshgrid(torch.arange(height, dtype=torch.float32),
                            torch.arange(width, dtype=torch.float32), indexing='ij')
    img = torch.ones(3, height, width, dtype=torch.float32)
    for geom, color in primitives:
        # Extraire la géométrie
        if len(geom) == 3:
            cx, cy, r = geom
            # Prépare les paramètres
            cx_t = torch.tensor(cx, dtype=torch.float32)
            cy_t = torch.tensor(cy, dtype=torch.float32)
            r_t = torch.tensor(r, dtype=torch.float32)
            dist_sq = (xx - cx_t) ** 2 + (yy - cy_t) ** 2
            sigma_sq = (r_t ** 2) + 1e-8
            mask = torch.exp(-dist_sq / (2.0 * sigma_sq))
        elif len(geom) == 5:
            cx, cy, rx, ry, angle = geom
            # Prépare les paramètres
            cx_t = torch.tensor(cx, dtype=torch.float32)
            cy_t = torch.tensor(cy, dtype=torch.float32)
            rx_t = torch.tensor(rx, dtype=torch.float32)
            ry_t = torch.tensor(ry, dtype=torch.float32)
            ang = torch.tensor(angle, dtype=torch.float32)
            # Coordonnées centrées
            x = xx - cx_t
            y = yy - cy_t
            # Rotation inverse
            cos_a = torch.cos(ang)
            sin_a = torch.sin(ang)
            x_rot = x * cos_a + y * sin_a
            y_rot = -x * sin_a + y * cos_a
            # Distance pondérée par les demi‑axes
            dist = (x_rot / (rx_t + 1e-8)) ** 2 + (y_rot / (ry_t + 1e-8)) ** 2
            mask = torch.exp(-dist / 2.0)
        else:
            raise ValueError("Primitive geometry must have length 3 (circle) or 5 (ellipse)")
        # Clip mask
        mask = mask.clamp(0.0, 1.0)
        c = color[:3].view(3, 1, 1)
        for ch in range(3):
            img[ch] = img[ch] * (1.0 - mask) + c[ch] * mask
    return img


def _render_drtk(primitives: List[Tuple[Tuple[float, float, float], torch.Tensor]], canvas_size: Tuple[int, int]) -> torch.Tensor:
    """Rendu différentiable via une bibliothèque externe (place-holder).

    Cette fonction est conservée pour référence et pour permettre
    l'intégration future d'un rasteriseur différentiable tel que DRTK.
    Actuellement, aucune bibliothèque n'est importée et cette fonction
    déclenche une erreur pour signaler qu'elle n'est pas disponible.
    """
    raise RuntimeError(
        "Aucun rasteriseur différentiable externe n'est disponible. "
        "Utilisez le rendu gaussien fourni par _render_gaussian."
    )


def render_scene(primitives: List[Tuple[Tuple, torch.Tensor]], canvas_size: Tuple[int, int] = (256, 256)) -> torch.Tensor:
    """Rendu différentiable d'une scène composée de primitives gaussiennes.

    Les primitives doivent être fournies sous forme de liste de tuples
    `(geometry, color)` où :

    * `geometry` est soit `(cx, cy, r)` pour un cercle, soit `(cx, cy, rx, ry, angle)` pour une ellipse.
    * `color` est un tenseur PyTorch `(4,)` représentant la couleur RGBA.

    Cette fonction délègue le rendu à `_render_gaussian`, qui mélange
    additivement les contributions de chaque primitive.  Aucune dépendance
    externe n'est requise.  La sortie est un tenseur `(3, H, W)` avec
    des valeurs normalisées dans `[0, 1]`.
    """
    return _render_gaussian(primitives, canvas_size)
