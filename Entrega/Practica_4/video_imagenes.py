import cv2
import os

# Ruta del video de entrada
ruta_video = "C:/Users/dante/Downloads/robos.mp4"

# Directorio de salida para los fotogramas
ruta_salida = "C:/Users/dante/Downloads/robos_imagenes"

# Crea el directorio de salida si no existe
if not os.path.exists(ruta_salida):
    os.makedirs(ruta_salida)

# Abre el video
cap = cv2.VideoCapture(ruta_video)

# Inicializa el contador de fotogramas
contador = 0

while True:
    # Lee el siguiente fotograma
    ret, frame = cap.read()

    if not ret:
        break

    # Redimensiona el fotograma a 28x28 píxeles
    frame_redimensionado = cv2.resize(frame, (28, 28))

    # Guarda el fotograma como una imagen JPG
    nombre_archivo = f'fotograma_{contador}.jpg'
    ruta_archivo = os.path.join(ruta_salida, nombre_archivo)
    cv2.imwrite(ruta_archivo, frame_redimensionado)

    contador += 1

# Libera el objeto de captura de video
cap.release()

print(f'Se han guardado {contador} fotogramas en {ruta_salida}')
