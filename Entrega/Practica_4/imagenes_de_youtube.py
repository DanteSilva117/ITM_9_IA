import cv2
from pytube import YouTube
import os

def download_youtube_video(youtube_url, output_folder):
    # Descargar el video de YouTube
    yt = YouTube(youtube_url)
    video_stream = yt.streams.filter(file_extension="mp4").first()
    video_stream.download(output_path=output_folder, filename="video.mp4")

def extract_frames(video_path, output_folder):
    # Cargar el video descargado
    cap = cv2.VideoCapture(video_path)

    # Crear la carpeta para guardar las imágenes
    os.makedirs(output_folder, exist_ok=True)

    # Leer los fotogramas y guardarlos como imágenes
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Redimensionar el fotograma a 28x28 píxeles
        resized_frame = cv2.resize(frame, (28, 28))

        # Guardar el fotograma como imagen .jpg
        image_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(image_filename, resized_frame)

        frame_count += 1

    # Liberar la captura de video
    cap.release()

    print(f"Se han guardado {frame_count} fotogramas como imágenes en formato .jpg en la carpeta {output_folder}.")

# Rutas personalizadas
youtube_url = "https://www.youtube.com/watch?v=lKDSt6SAeuM&ab_channel=ReedTimmer"
video_output_folder = "E:/ITM_9_IA/Entrega/Practica_4/video"
image_output_folder = "E:/ITM_9_IA/Entrega/Practica_4/imagenes_video"

# Descargar el video y extraer los fotogramas
download_youtube_video(youtube_url, video_output_folder)
extract_frames(os.path.join(video_output_folder, "video.mp4"), image_output_folder)
