import os
from pytube import YouTube

def download_youtube_video(youtube_url, output_folder):
    try:
        # Crear la carpeta de salida si no existe
        os.makedirs(output_folder, exist_ok=True)

        # Descargar el video de YouTube
        yt = YouTube(youtube_url)
        video_stream = yt.streams.filter(file_extension="mp4").first()
        video_stream.download(output_path=output_folder)

        print(f"El video se ha descargado correctamente en {output_folder}.")
    except Exception as e:
        print(f"Error al descargar el video: {e}")

# URL del video de YouTube
youtube_url = "https://www.youtube.com/watch?v=GoSyZX_Dj7E&ab_channel=calamar2producciones"

# Ruta de la carpeta donde deseas guardar el video
output_folder = "E:/ITM_9_IA/Entrega/Practica_4/video"

# Descargar el video
download_youtube_video(youtube_url, output_folder)
