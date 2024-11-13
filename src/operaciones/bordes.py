import numpy as np
from src.operaciones import filterImage, gaussianFilter
from src.operaciones import adjustIntensity

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

    if sigma <= 0:
        raise ValueError('El valor de sigma debe ser mayor que 0')

    N = 2 * int(np.ceil(3 * sigma)) + 1
    center = N // 2

    kernel = np.zeros((N, N), dtype=np.float64)

    for x in range(N):
        for y in range(N):
            # Coordenadas relativas al centro
            x_dist = x - center
            y_dist = y - center
            # Fórmula de LoG
            kernel[x, y] = ((x_dist**2 + y_dist**2 - sigma**2) / sigma**4) * np.exp(-(x_dist**2 + y_dist**2) / (2 * sigma**2))
    
    # Normalizar el kernel para que la suma sea aproximadamente cero
    kernel -= kernel.mean()
    #kernel /= np.sum(kernel)

    #Convolucionar la imagen con el kernel calculado, sombrero mexicano
    outImage = filterImage(inImage, kernel)

    return outImage


'''
Implementar el detector de bordes de Canny.
No renormaliza

outImage = edgeCanny (inImage, sigma, tlow, thigh)
    inImage, outImage: ...
    sigma: Parámetro sigma del filtro Gaussiano.
    tlow, thigh: Umbrales de histéresis bajo y alto, respectivamente.

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


    #Bucle con indices adecuados para no situarse en los bordes de la imagen
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            
            #Determinar cuales son los vecinos segun la direccion de la normal
            if (0 <= direction[i,j] < 22.5) or (157.5 <= direction[i,j] <= 180):
                neighbors = (magnitude[i, j + 1], magnitude[i, j - 1])  
            elif (22.5 <= direction[i,j] < 67.5):
                neighbors = (magnitude[i - 1, j - 1], magnitude[i + 1, j + 1])
            elif (67.5 <= direction[i,j] < 112.5):
                neighbors = (magnitude[i + 1, j], magnitude[i - 1, j])
            elif (112.5 <= direction[i,j] < 157.5):
                neighbors = (magnitude[i + 1, j - 1], magnitude[i - 1, j + 1])

            #No es 0 si >= que el maximo de sus vecinos, que son los que estan en la direccion de la normal
            if magnitude[i, j] >= max(neighbors):
                nms_image[i, j] = magnitude[i, j]

    #Umbralización por histeresis

    #Obtener los bordes debiles
    weak_edges = ((nms_image > tlow) & (nms_image <= thigh))

    #Marcas bordes fuertes
    outImage = (nms_image > thigh).astype(np.float64) #Float64 para seguir patron de salida

    #Visitados
    visited = np.zeros_like(nms_image, dtype=bool)

    #Coordenadas para mirar alrededor de una coordenada
    surrounding = [(-1, -1), (-1, 0), (-1, 1),(0, -1),(0, 1),(1, -1), (1, 0), (1, 1)]

    while True:

        prev_outImage = outImage.copy()

        #Conectar bordes débiles a bordes fuertes
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if visited[i,j]: #Si ya esta visitado continuar
                    continue

                if outImage[i,j] == 1.0: #Si es un borde fuerte
                    visited[i,j] = True

                    for offset_i,  offset_j in surrounding:
                        ni, nj = i + offset_i, j + offset_j

                        #Si la coordenada vecina es un borde debil marcar esa coordenada como borde fuerte
                        if weak_edges[ni, nj] == 1:
                            outImage[ni, nj] = 1.0

        #Para de iterar cuando no hay mas cambios iterando
        if np.array_equal(outImage, prev_outImage):
            break
        

    return outImage
