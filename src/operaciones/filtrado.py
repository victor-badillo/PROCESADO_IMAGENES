import numpy as np

'''
Función para realizar un filtro espacial mediante convolución sobre una imagen con un 
kernel arbitrario que se le pasa como parámetro
No renormaliza
'''
def filterImage(inImage, kernel):

    # Obtener el tamaño del kernel
    P, Q = kernel.shape
    
    # Calcular el padding necesario
    pad_y = P // 2
    pad_x = Q // 2

    # Añadir padding a la imagen utilizando el modo 'reflect'
    paddedImage = np.pad(inImage, ((pad_y, pad_y), (pad_x, pad_x)), mode='reflect')

    # Crear una imagen de salida inicializada a ceros
    outImage = np.zeros_like(inImage)

    # Realizar la convolución
    for i in range(inImage.shape[0]):
        for j in range(inImage.shape[1]):
            # Extraer la región de interés
            region = paddedImage[i:i + P, j:j + Q]
            # Aplicar el kernel
            outImage[i, j] = np.sum(region * kernel)

    return outImage


'''
Función que calcula un kernel Gaussiano unidimensional
'''
def gaussKernel1D(sigma):
    # Calcular N a partir de sigma
    N = 2 * int(np.ceil(3 * sigma)) + 1
    
    # Crear un vector para el kernel
    kernel = np.zeros(N)
    
    # Calcular el centro del kernel
    center = N // 2
    
    # Calcular los valores del kernel
    for x in range(N):
        # Usar la fórmula de la gaussiana
        kernel[x] = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    
    kernel /= np.sum(kernel)

    return kernel

'''
#Version optimizada que utliza numpy, preguntar al profesor si se puede utilizar
def gaussKernel1D(sigma):
    # Calcular N a partir de sigma
    N = 2 * int(np.ceil(3 * sigma)) + 1
    
    # Crear un vector de índices centrados en 0
    x = np.arange(N) - N // 2
    
    # Calcular el kernel gaussiano usando NumPy (sin bucles)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    
    # Normalizar el kernel para que la suma sea 1
    kernel /= np.sum(kernel)

    return kernel
'''

'''
Función que permite realizar un suavizado Gaussiano bidimensional usando un filtro NxN de 
parámetro sigma.
No renormaliza
'''
def gaussianFilter(inImage, sigma):
    # Obtener el kernel gaussiano unidimensional
    kernel_1d = gaussKernel1D(sigma)
    
    #-1 para que numpy detecte automaticamente el numero de la dimension
    # Aplicar la convolución con el kernel 1xN
    intermediate_result = filterImage(inImage, kernel_1d.reshape(1, -1))
    
    # Aplicar la convolución con el kernel transpuesto N×1
    outImage = filterImage(intermediate_result, kernel_1d.reshape(-1, 1))
    
    return outImage


'''
Función que implementa el filtro de medianas bidimensional, especificando el tamaño del filtro
No renormaliza
'''
def medianFilter(inImage, filterSize):

    # Calcular el padding necesario
    pad = filterSize // 2

    # Añadir padding a la imagen utilizando el modo 'reflect'
    paddedImage = np.pad(inImage, ((pad, pad), (pad, pad)), mode='reflect')

    # Crear una imagen de salida inicializada a ceros
    outImage = np.zeros_like(inImage)

    # Obtener las dimensiones de la imagen de entrada
    rows, cols = inImage.shape

    # Aplicar el filtro de mediana
    for i in range(rows):
        for j in range(cols):
            # Extraer la región de interés (ventana de tamaño filterSize x filterSize)
            region = paddedImage[i:i + filterSize, j:j + filterSize]
            # Calcular la mediana y asignarla al píxel correspondiente en la imagen de salida
            outImage[i, j] = np.median(region)

    return outImage
