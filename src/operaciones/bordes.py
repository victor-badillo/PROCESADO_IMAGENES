import numpy as np
from src.operaciones import filterImage, gaussianFilter

'''
Implementar una función que permita obtener las componentes Gx y Gy del gradiente de una
imagen, pudiendo elegir entre los operadores de Roberts, CentralDiff (Diferencias centrales
de Prewitt/Sobel sin promedio: i.e. [-1, 0, 1] y transpuesto), Prewitt y Sobel.
No renormaliza

[gx, gy] = gradientImage (inImage, operator)
    inImage: ...
    gx, gy: Componentes Gx y Gy del gradiente.
    operator: Permite seleccionar el operador utilizado mediante los valores: 'Roberts',
        'CentralDiff', 'Prewitt' o 'Sobel'.
'''
def gradientImage(inImage, operator):

    if operator == 'Roberts':
        Gx_kernel = np.array([[-1, 0], [0, 1]], dtype=np.float64)
        Gy_kernel = np.array([[0, -1], [1, 0]], dtype=np.float64)

    elif operator == 'CentralDiff':
        Gx_kernel = np.array([[-1, 0, 1]], dtype=np.float64)
        Gy_kernel = np.array([[-1], [0], [1]], dtype=np.float64)

    elif operator == 'Prewitt':
        Gx_kernel = np.array([[-1, 0, 1],
                              [-1, 0, 1],
                              [-1, 0, 1]], dtype=np.float64)
        Gy_kernel = np.array([[-1, -1, -1],
                              [0, 0, 0],
                              [1, 1, 1]], dtype=np.float64)

    elif operator == 'Sobel':
        Gx_kernel = np.array([[-1, 0, 1],
                              [-2, 0, 2],
                              [-1, 0, 1]], dtype=np.float64)
        Gy_kernel = np.array([[-1, -2, -1],
                              [0, 0, 0],
                              [1, 2, 1]], dtype=np.float64)

    else:
        raise ValueError("Operador no válido. Debe ser 'Roberts', 'CentralDiff', 'Prewitt' o 'Sobel'.")

    gx = filterImage(inImage, Gx_kernel)
    gy = filterImage(inImage, Gy_kernel)

    return gx, gy


'''
Implementar el filtro Laplaciano de Gaussiano que permita especificar el parámetro sigma de la
Gaussiana utilizada.
No renormaliza

outImage = LoG (inImage, sigma)
    inImage, outImage: ...
    sigma: Parámetro sigma de la Gaussiana
'''
def LoG(inImage, sigma):
    
    #Suavizar la imagen con un filtro gaussiano
    smoothedImage = gaussianFilter(inImage, sigma)
    
    #Kernel laplaciano
    laplacianKernel = np.array([[0, 1, 0],
                                [1, -4, 1],
                                [0, 1, 0]], dtype=np.float64)
    
    #Filtrar imagen suavizada con kernel laplaciano
    outImage = filterImage(smoothedImage, laplacianKernel)
    
    return outImage



'''
Implementar el detector de bordes de Canny.
No renormaliza

outImage = edgeCanny (inImage, sigma, tlow, thigh)
    inImage, outImage: ...
    sigma: Parámetro sigma del filtro Gaussiano.
    tlow, thigh: Umbrales de histéresis bajo y alto, respectivamente.

'''
'''
def edgeCanny(inImage, sigma, tlow, thigh):
    
    #Suavizar la imagen con un filtro gaussiano
    smoothed_image =  gaussianFilter(inImage, sigma)

    #Calcular los gradientes
    gx, gy = gradientImage(smoothed_image, 'Sobel')
    magnitude = np.sqrt(gx**2 + gy**2)             #Magnitud
    direction = np.degrees(np.arctan2(gy, gx))     #Direccion en grados y rango [0,180]
    direction = np.where(direction < 0, direction + 180, direction)

    #Supresion de no maximos
    nms_image = np.zeros_like(magnitude, dtype=np.float64)
    rows, cols = magnitude.shape

    #Bucle con indices adecuados para no situarse en los bordes de la imagen
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            
            #Determinar cuales son los vecinos segun la direccion de la normal
            if (0 <= direction < 22.5) or (157.5 <= direction <= 180):
                neighbors = (magnitude[i, j + 1], magnitude[i, j - 1])
            elif (22.5 <= direction < 67.5):
                neighbors = (magnitude[i + 1, j - 1], magnitude[i - 1, j + 1])
            elif (67.5 <= direction < 112.5):
                neighbors = (magnitude[i + 1, j], magnitude[i - 1, j])
            elif (112.5 <= direction < 157.5):
                neighbors = (magnitude[i - 1, j - 1], magnitude[i + 1, j + 1])

            #No es 0 si >= que el maximo de sus vecinos, que son los que estan en la direccion de la normal
            if magnitude[i, j] >= max(neighbors):
                nms_image[i, j] = magnitude[i, j]

    #Umbralización por histeresis
    strong_edges = (nms_image > thigh)                          #Fuertes los que son mayores que el umbral superior
    weak_edges = ((nms_image >= tlow) & (nms_image <= thigh))   #Debiles los que son menores que el umbral superior y mayores que el umbral inferior

    outImage = np.zeros_like(nms_image, dtype=np.uint8)

    #Marcas bordes fuertes
    outImage[strong_edges] = 1

    #Conectar bordes débiles a bordes fuertes
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if weak_edges[i, j]:
                if ((outImage[i + 1, j] == 1) or (outImage[i - 1, j] == 1) or
                    (outImage[i, j + 1] == 1) or (outImage[i, j - 1] == 1) or
                    (outImage[i + 1, j + 1] == 1) or (outImage[i - 1, j - 1] == 1) or
                    (outImage[i + 1, j - 1] == 1) or (outImage[i - 1, j + 1] == 1)):
                    outImage[i, j] = 1


    outImage = outImage.astype(inImage.dtype)

    return outImage
'''

def edgeCanny(inImage, sigma, tlow, thigh):

    if(thigh < tlow):
        raise ValueError("El umbral superior de histéresis debe ser superior al umbral inferior")
    
    #Suavizar la imagen con un filtro gaussiano
    smoothed_image =  gaussianFilter(inImage, sigma)

    #Calcular los gradientes
    gx, gy = gradientImage(smoothed_image, 'Sobel')
    magnitude = np.sqrt(gx**2 + gy**2)             #Magnitud
    direction = np.degrees(np.arctan2(gy, gx))     #Direccion en grados y rango [0,180]
    direction = np.where(direction < 0, direction + 180, direction)

    #Supresion de no maximos
    nms_image = np.zeros_like(magnitude, dtype=np.float64)
    rows, cols = magnitude.shape

    perp = np.zeros((rows, cols, 2), dtype=int) #Almacenamiento de las perpendiculares a la normal

    #Bucle con indices adecuados para no situarse en los bordes de la imagen
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            
            #Determinar cuales son los vecinos segun la direccion de la normal
            if (0 <= direction[i,j] < 22.5) or (157.5 <= direction[i,j] <= 180):
                neighbors = (magnitude[i, j + 1], magnitude[i, j - 1])  
                perp[i,j] = (0, 1)  #Horizontal
            elif (22.5 <= direction[i,j] < 67.5):
                neighbors = (magnitude[i + 1, j - 1], magnitude[i - 1, j + 1])
                perp[i,j] = (1, -1)  #Diagonal /
            elif (67.5 <= direction[i,j] < 112.5):
                neighbors = (magnitude[i + 1, j], magnitude[i - 1, j])
                perp[i,j] = (1, 0)  #Vertical
            elif (112.5 <= direction[i,j] < 157.5):
                neighbors = (magnitude[i - 1, j - 1], magnitude[i + 1, j + 1])
                perp[i,j] = (-1, -1)  #Diagonal \

            #No es 0 si >= que el maximo de sus vecinos, que son los que estan en la direccion de la normal
            if magnitude[i, j] >= max(neighbors):
                nms_image[i, j] = magnitude[i, j]

    #Umbralización por histeresis
    strong_edges = (nms_image > thigh)                          #Fuertes los que son mayores que el umbral superior
    weak_edges = ((nms_image >= tlow) & (nms_image <= thigh))   #Debiles los que son menores que el umbral superior y mayores que el umbral inferior

    outImage = np.zeros_like(nms_image, dtype=np.uint8)

    #Marcas bordes fuertes
    outImage[strong_edges] = 1

    #Visitados
    visited = np.zeros_like(nms_image, dtype=bool)

    #Conectar bordes débiles a bordes fuertes
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if visited[i,j]: #Si ya esta visitado continuar
                continue

            if outImage[i,j] == 1: #Si es un borde
                
                perp_i, perp_j = perp[i,j]    #Obtener perpendicular a la normal

                
                current_i, current_j = i+perp_i, j+perp_j #Indices de recorrido
                #Recorrer en una direccion de la perpendicular
                while weak_edges[current_i, current_j] == 1 and (visited[current_i, current_j] == False) and 0 <= current_i < rows and 0 <= current_j < cols:
                    outImage[current_i, current_j] = 1
                    visited[current_i, current_j] = True
                    current_i = current_i + perp_i
                    current_j = current_j + perp_j

                current_i, current_j = i-perp_i, j-perp_j #Indices de recorrido
                #Recorrer en la otra direccion de la perpendicular
                while weak_edges[current_i, current_j] == 1 and (visited[current_i, current_j] == False) and 0 <= current_i < rows and 0 <= current_j < cols:
                    outImage[current_i, current_j] = 1
                    visited[current_i, current_j] = True
                    current_i = current_i - perp_i
                    current_j = current_j - perp_j


    outImage = outImage.astype(inImage.dtype)

    return outImage
