import numpy as np

'''
Función para realizar un filtro espacial mediante convolución sobre una imagen con un 
kernel arbitrario que se le pasa como parámetro
No renormaliza

outImage = filterImage (inImage, kernel)
    inImage, outImage: ...
    kernel: Matriz PxQ con el kernel del filtro de entrada. Se asume que la posición central
        del filtro está en (⌊P/2⌋ + 1, ⌊Q/2⌋ + 1).
'''
def filterImage(inImage, kernel):

    P, Q = kernel.shape
    
    #Calcular el padding
    pad_y = P // 2
    pad_x = Q // 2

    #Añadir padding con modo reflect
    paddedImage = np.pad(inImage, ((pad_y, pad_y), (pad_x, pad_x)), mode='reflect')

    outImage = np.zeros_like(inImage)

    #Convolucion
    for i in range(inImage.shape[0]):
        for j in range(inImage.shape[1]):
            
            region = paddedImage[i:i + P, j:j + Q]  #Region
            outImage[i, j] = np.sum(region * kernel)

    return outImage


'''
Función que calcula un kernel Gaussiano unidimensional con sigma dado

kernel = gaussKernel1D (sigma)
    sigma: Parámetro sigma de entrada.
    kernel: Vector 1xN con el kernel de salida, teniendo en cuenta que:
        • El centro x = 0 de la Gaussiana está en la posición ⌊N/2⌋ + 1.
        • N se calcula a partir de sigma como N = 2⌈3sigma⌉ + 1.
'''
def gaussKernel1D(sigma):

    N = 2 * int(np.ceil(3 * sigma)) + 1
    kernel = np.zeros(N)
    center = N // 2
    
    # Calcular los valores del kernel
    for x in range(N):

        kernel[x] = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    
    kernel /= np.sum(kernel)    #Asi no cambio el brillo de la imagen

    return kernel



'''
Función que permite realizar un suavizado Gaussiano bidimensional usando un filtro NxN de 
parámetro sigma.
No renormaliza

outImage = gaussianFilter (inImage, sigma)
    inImage, outImage, sigma: ...
    Nota. Como el filtro Gaussiano es lineal y separable podemos implementar este suavi-
    zado simplemente convolucionando la imagen, primero, con un kernel Gaussiano unidi-
    mensional 1xN y, luego, convolucionando el resultado con el kernel transpuesto N x 1.
'''
def gaussianFilter(inImage, sigma):
    
    kernel_1d = gaussKernel1D(sigma)    #Calcular kernel

    #Convolución con el kernel 1xN
    intermediate_result = filterImage(inImage, kernel_1d.reshape(1, -1))    #-1 para que numpy detecte automaticamente el numero de la dimension
    
    #Convolución con el kernel transpuesto N×1
    outImage = filterImage(intermediate_result, kernel_1d.reshape(-1, 1))
    
    return outImage


'''
Función que implementa el filtro de medianas bidimensional, especificando el tamaño del filtro
No renormaliza

outImage = medianFilter (inImage, filterSize)
    inImage, outImage: ...
    filterSice: Valor entero N indicando que el tamaño de ventana es de NxN. La posición
        central de la ventana es (⌊N/2⌋ + 1, ⌊N/2⌋ + 1).
'''
def medianFilter(inImage, filterSize):

    if filterSize <= 0:
        raise ValueError("filterSize debe ser mayor que 0")

    pad = filterSize // 2

    #Añadir padding con modo reflect
    paddedImage = np.pad(inImage, ((pad, pad), (pad, pad)), mode='reflect')

    outImage = np.zeros_like(inImage)

    rows, cols = inImage.shape  #Obtener las dimensiones de la imagen de entrada

    #Calcular valor para cada pixel
    for i in range(rows):
        for j in range(cols):
            region = paddedImage[i:i + filterSize, j:j + filterSize]
            outImage[i, j] = np.median(region)

    return outImage
