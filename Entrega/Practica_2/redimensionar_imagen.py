from PIL import Image
import os

def redimensionar_imagenes(ruta_origen, ruta_destino, tamano_deseado=(50, 50)):
    # Asegúrate de que la carpeta de destino exista
    if not os.path.exists(ruta_destino):
        os.makedirs(ruta_destino)

    # Recorre todos los archivos en la carpeta de origen
    for archivo in os.listdir(ruta_origen):
        ruta_archivo_origen = os.path.join(ruta_origen, archivo)
        ruta_archivo_destino = os.path.join(ruta_destino, archivo)

        # Abre la imagen
        imagen = Image.open(ruta_archivo_origen)

        # Redimensiona la imagen
        imagen_redimensionada = imagen.resize(tamano_deseado, Image.BILINEAR)  # Cambio aquí

        # Guarda la imagen redimensionada en la carpeta de destino
        imagen_redimensionada.save(ruta_archivo_destino)

if __name__ == "__main__":
    ruta_origen = "E:/ITM_9_IA/Entrega/Practica_2/wally/n"  # Cambia esto a la ruta de tus imágenes
    ruta_destino = "E:/ITM_9_IA/Entrega/Practica_2/wally/n"  # Cambia esto a la ruta deseada

    redimensionar_imagenes(ruta_origen, ruta_destino)
    print("Imágenes redimensionadas correctamente.")

