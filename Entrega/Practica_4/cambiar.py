from PIL import Image
import os

def convertir_y_eliminar_png(directorio):
    # Obtener lista de todos los archivos en el directorio
    archivos = os.listdir(directorio)
    
    for archivo in archivos:
        # Comprobar si el archivo es un .png
        if archivo.lower().endswith('.png'):
            ruta_png = os.path.join(directorio, archivo)
            ruta_jpg = os.path.join(directorio, archivo[:-4] + '.jpg')
            
            # Abrir la imagen y convertirla a .jpg
            with Image.open(ruta_png) as imagen:
                imagen = imagen.convert('RGB')  # Asegurarse de que la imagen está en modo RGB antes de guardarla como .jpg
                imagen.save(ruta_jpg, 'JPEG')
            
            # Eliminar el archivo .png
            os.remove(ruta_png)
            print(f'Convertido y eliminado: {archivo}')

# Especificar el directorio donde están las imágenes .png
directorio_imagenes = 'E://ITM_9_IA//Entrega//Practica_4//Imagenes//inundacion'

# Llamar a la función para convertir y eliminar los archivos .png
convertir_y_eliminar_png(directorio_imagenes)
