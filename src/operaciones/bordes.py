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
    # Paso 1: Suavizar la imagen utilizando el filtro gaussiano
    smoothedImage = gaussianFilter(inImage, sigma)
    
    # Paso 2: Definir el kernel Laplaciano
    laplacianKernel = np.array([[0, 1, 0],
                                [1, -4, 1],
                                [0, 1, 0]], dtype=np.float64)
    
    # Paso 3: Aplicar la convolución de la imagen suavizada con el kernel Laplaciano
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
def edgeCanny(inImage, sigma, tlow, thigh):
    # Paso 1: Suavizar la imagen utilizando el filtro gaussiano
    smoothedImage = gaussianFilter(inImage, sigma)

    # Paso 2: Calcular los gradientes en las direcciones x e y
    gx, gy = gradientImage(smoothedImage, 'Sobel')

    # Paso 3: Calcular la magnitud y dirección del gradiente
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    angle = np.arctan2(gy, gx) * (180.0 / np.pi)  # Convertir a grados
    angle = (angle + 180) % 180  # Ajustar el rango a [0, 180]

    # Paso 4: Supresión de no-máximos
    nmsImage = np.zeros_like(magnitude, dtype=np.uint8)
    rows, cols = magnitude.shape

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # Redondear el ángulo a 0, 45, 90 o 135 grados
            q = 255
            r = 255

            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            elif 22.5 <= angle[i, j] < 67.5:
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]
            elif 67.5 <= angle[i, j] < 112.5:
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]
            elif 112.5 <= angle[i, j] < 157.5:
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]

            # Supresión de no-máximos
            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                nmsImage[i, j] = magnitude[i, j]
            else:
                nmsImage[i, j] = 0

    # Paso 5: Umbralización con histeresis
    strongEdges = (nmsImage >= thigh)
    weakEdges = (nmsImage >= tlow) & (nmsImage < thigh)

    # Crear una imagen de salida
    outImage = np.zeros_like(nmsImage, dtype=np.uint8)

    # Conectar bordes fuertes y débiles
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if strongEdges[i, j]:
                outImage[i, j] = 255
            elif weakEdges[i, j]:
                # Comprobar si hay bordes fuertes en los vecinos
                if ((strongEdges[i + 1, j] or strongEdges[i - 1, j] or
                     strongEdges[i, j + 1] or strongEdges[i, j - 1] or
                     strongEdges[i + 1, j + 1] or strongEdges[i - 1, j - 1] or
                     strongEdges[i + 1, j - 1] or strongEdges[i - 1, j + 1])):
                    outImage[i, j] = 255

    return outImage
