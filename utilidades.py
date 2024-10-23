import cv2
import numpy as np
from configuracion import INPUT_IMAGES, OUTPUT_IMAGES
import matplotlib.pyplot as plt

'''
Funcion para cargar imagenes en escala de grises y float64
'''
def load_image(nombre_imagen):

    img_path = INPUT_IMAGES + nombre_imagen
   
    inImage = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)    #Escala de grises
    
    if inImage is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {img_path}")
    
    #Convertir a tipo float64 y normalizar al rango [0, 1]
    inImage = inImage.astype(np.float64) / 255.0
    
    return inImage


'''
Funcion para visualizar imagen, imagen [0,255]
'''
def visualize_image_int(title, image):
    
    cv2.imshow(title, (image * 255).astype(np.uint8) )
    cv2.waitKey(0) 
    cv2.destroyAllWindows()


'''
Funcion para visualizar imagen, imagen [0,1]
'''
def visualize_image_float(title, image):
    
    cv2.imshow(title, image )
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''
Funcion para guardar imagen, imagen [0,255]
'''
def save_image_int(nombre_imagen, image):

    img_path = OUTPUT_IMAGES + nombre_imagen
    success = cv2.imwrite(img_path,(image * 255).astype(np.uint8))

    if success:
        print("La imagen ==> " + nombre_imagen + " ==> se guardó correctamente.")
    else:
        print("Error al guardar la imagen ==>" + img_path)


'''
Funcion para guardar imagen, imagen [0,1]
'''
def save_image_float(nombre_imagen, image):

    img_path = OUTPUT_IMAGES + nombre_imagen
    success = cv2.imwrite(img_path,image)

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

'''
Funcion para plotear el histograma de una imagen
'''
def plot_histogram(hist, bin_edges, title='Histogram'):
    plt.figure(figsize=(8, 6))
    plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor='black', align='edge')
    plt.title(title)
    plt.xlabel('Pixel intensity')
    plt.ylabel('Frequency')
    plt.show()