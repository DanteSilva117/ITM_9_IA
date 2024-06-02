import os

def rename_images(folder_path):
    # Verifica si la ruta de la carpeta existe
    if not os.path.exists(folder_path):
        print(f"La carpeta {folder_path} no existe.")
        return
    
    # Obtiene la lista de archivos en la carpeta
    files = os.listdir(folder_path)
    
    # Filtra solo los archivos de imagen (puedes ajustar esto según tus necesidades)
    image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    
    # Renombra las imágenes
    for i, image_file in enumerate(image_files, start=1):
        new_name = f"asalto{i}.jpg"  # Cambia la extensión según el tipo de archivo
        old_path = os.path.join(folder_path, image_file)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
        print(f"Renombrando {image_file} a {new_name}")
    
    print("¡Todas las imágenes han sido renombradas!")

# Ruta de la carpeta donde están las imágenes
ruta_carpeta = "E:/ITM_9_IA/Entrega/Practica_4/Imagenes/asalto/"
rename_images(ruta_carpeta)
