import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path
import argparse

# Ré-utilisez l'architecture du générateur du fichier d'entraînement
from next_frame_gan import GeneratorUNet

def generate_sequence(args: argparse.Namespace):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Charger le modèle Générateur entraîné
    model = GeneratorUNet().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print("Modèle générateur chargé.")

    # 2. Préparer les transformations d'image
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    def denorm(img_tensor):
        return (img_tensor * 0.5 + 0.5).clamp(0, 1)

    # 3. Charger les images de départ
    try:
        img1 = Image.open(args.start_frame_1).convert("RGB")
        # Si une seule frame est fournie, on la duplique.
        # Le modèle s'attend à un mouvement, donc il prédira un mouvement minimal.
        img2_path = args.start_frame_2 if args.start_frame_2 else args.start_frame_1
        img2 = Image.open(img2_path).convert("RGB")
    except FileNotFoundError as e:
        print(f"Erreur: Fichier image non trouvé - {e}")
        return

    tensor1 = transform(img1).to(device)
    tensor2 = transform(img2).to(device)

    # 4. Boucle de génération
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Sauvegarder les frames de départ
    img1.save(output_dir / "frame_000.png")
    img2.save(output_dir / "frame_001.png")

    print(f"Génération de {args.num_frames} frames...")
    with torch.no_grad():
        for i in range(args.num_frames):
            # Concaténer les deux dernières frames pour prédire la suivante
            input_tensor = torch.cat([tensor1, tensor2], dim=0).unsqueeze(0) # Ajouter une dimension de batch
            
            # Prédiction
            predicted_tensor = model(input_tensor).squeeze(0) # Enlever la dimension de batch
            
            # Convertir le tenseur en image et sauvegarder
            predicted_img_tensor = denorm(predicted_tensor.cpu())
            predicted_img = transforms.ToPILImage()(predicted_img_tensor)
            predicted_img.save(output_dir / f"frame_{i+2:03d}.png")
            
            print(f"Frame {i+2} générée.")

            # Mettre à jour les tenseurs pour la prochaine itération
            # La frame T devient la nouvelle frame T-1
            # La prédiction devient la nouvelle frame T
            tensor1 = tensor2
            tensor2 = predicted_tensor
            
    print(f"Séquence sauvegardée dans le dossier '{output_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Générer une séquence de frames à partir d'un modèle GAN entraîné.")
    parser.add_argument("--model_path", type=str, required=True, help="Chemin vers le modèle générateur (.pth).")
    parser.add_argument("--start_frame_1", type=str, required=True, help="Chemin vers la première image de la séquence.")
    parser.add_argument("--start_frame_2", type=str, help="(Optionnel) Chemin vers la deuxième image. Si non fourni, la première est utilisée deux fois.")
    parser.add_argument("--num_frames", type=int, default=10, help="Nombre de frames à générer.")
    parser.add_argument("--image_size", type=int, default=256, help="Taille des images sur lesquelles le modèle a été entraîné.")
    parser.add_argument("--output_dir", type=str, default="generated_sequence", help="Dossier pour sauvegarder la séquence générée.")
    
    args = parser.parse_args()
    generate_sequence(args)