import os
from PIL import Image

# Ruta de la carpeta con las imágenes
ruta_carpeta = "C:/Users/dante/Downloads/Nueva carpeta"

# Lista de archivos en la carpeta
archivos = os.listdir(ruta_carpeta)

# Redimensiona cada imagen en la carpeta
for archivo in archivos:
    if archivo.lower().endswith((".jpg", ".jpeg", ".png")):
        ruta_imagen = os.path.join(ruta_carpeta, archivo)
        imagen = Image.open(ruta_imagen)
        imagen_redimensionada = imagen.resize((28, 28))
        ruta_imagen_redimensionada = os.path.join(ruta_carpeta, f"redimensionada_{archivo}")
        imagen_redimensionada.save(ruta_imagen_redimensionada)
        print(f"Imagen redimensionada guardada en: {ruta_imagen_redimensionada}")

        # Elimina la imagen original si no tiene las dimensiones correctas
        if imagen.size != (50, 50):
            os.remove(ruta_imagen)

print("Proceso completado.")
