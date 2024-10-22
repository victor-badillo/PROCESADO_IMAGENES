import cv2
from skimage import io
import numpy as np
from configuracion import INPUT_IMAGES, OUTPUT_IMAGES

def load_image(nombre_imagen):

    img_path = INPUT_IMAGES + nombre_imagen
    #Cargar la imagen en escala de grises
    inImage = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if inImage is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {img_path}")
    
    #Convertir a tipo de dato flotante y normalizar al rango [0, 1]
    inImage = inImage.astype(np.float64) / 255.0
    
    return inImage


def visualizar_imagen_int(title, image):
    
    cv2.imshow(title, (image * 255).astype(np.uint8) )
    cv2.waitKey(0) 
    cv2.destroyAllWindows()


def visualizar_imagen_float(title, image):
    
    cv2.imshow(title, image )
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#CREO QUE TENGO QUE CAMBIARLA Y GUARDAR EN RANGO [0, 1]
def guardar_imagen_int(nombre_imagen, image):

    img_path = OUTPUT_IMAGES + nombre_imagen
    success = cv2.imwrite(img_path,(image * 255).astype(np.uint8))  # Guardar la imagen ajustada

    if success:
        print("La imagen ==> " + nombre_imagen + " ==> se guardó correctamente.")
    else:
        print("Error al guardar la imagen ==>" + img_path)


def guardar_imagen_float(nombre_imagen, image):

    img_path = OUTPUT_IMAGES + nombre_imagen
    success = cv2.imwrite(img_path,image)  # Guardar la imagen ajustada

    if success:
        print("La imagen ==> " + nombre_imagen + " ==> se guardó correctamente.")
    else:
        print("Error al guardar la imagen ==>" + img_path)


'''
Función para calcular el histograma de una imagen de forma manual
'''
def histogram(inImage, bins, min_range=0.0, max_range=1.0):

    hist = np.zeros(bins, dtype=int) #Histograma con 0

    bin_size = (max_range - min_range) / bins   #Tamaño del bin

    for pixel in inImage.flatten():
        
        if min_range <= pixel <= max_range: #Omitir pixeles con valores fuera del rango
            bin_index = int((pixel - min_range) // bin_size)
            if bin_index == bins:  #Caso en el que pixel == max_range
                bin_index = bins - 1
            hist[bin_index] += 1

    bin_edges = np.linspace(min_range, max_range, bins + 1) #Calcular los límites de los bins, +1 para limite superior

    return hist, bin_edges

