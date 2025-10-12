import os
import subprocess

# === 1. Téléchargement avec yt-dlp ===
url = "https://www.youtube.com/watch?v=vRPCAAUBMms"  # <-- remplace par le lien de ta vidéo
output_folder = "video_download"
os.makedirs(output_folder, exist_ok=True)

video_path = os.path.join(output_folder, "video.mp4")

print("Téléchargement en cours avec yt-dlp...")
subprocess.run([
    "yt-dlp",
    "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
    "-o", video_path,
    url
])
print(f"✅ Vidéo téléchargée : {video_path}")

# === 2. Extraction des frames optimisée pour les animés ===
frames_folder = "frames_output"
os.makedirs(frames_folder, exist_ok=True)
output_pattern = os.path.join(frames_folder, "frame_%04d.png")

# Commande optimisée pour les animés
scene_threshold = "0.05"  # tu peux mettre 0.03 à 0.08 selon ta sensibilité
cmd = [
    "ffmpeg",
    "-i", video_path,
    "-vf", f"hqdn3d=2:1:2:1,select='gt(scene,{scene_threshold})',setpts=N/FRAME_RATE/TB",
    "-vsync", "vfr",
    output_pattern
]

print("Extraction des images en cours...")
subprocess.run(cmd)
print(f"✅ Extraction terminée ! Les images sont dans : {frames_folder}")