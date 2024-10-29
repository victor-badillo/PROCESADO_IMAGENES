import numpy as np
from configuracion import DEFAULT_CENTER, DEFAULT_SE

'''
outImage = erode (inImage, SE, center=[])
    inImage, outImage: ...
    SE: Matriz PxQ de zeros y unos definiendo el elemento estructurante.
    center: Vector 1x2 con las coordenadas del centro de SE. Se asume que el [0 0] es
        la esquina superior izquierda. Si es un vector vacío (valor por defecto), el centro
        se calcula como (⌊P/2⌋ + 1, ⌊Q/2⌋ + 1).
'''
def erode(inImage, SE, center=DEFAULT_CENTER):

    #Verificar que el SE tiene dimensiones validas
    if SE.shape[0] < 1 or SE.shape[1] < 1:
        raise ValueError("El SE debe tener al menos dimension 1x1")
    
    #Calcular centro si no se especifica uno
    if not center:
        center = [SE.shape[0] // 2, SE.shape[1] // 2]

    #Verificar que el centro no se sale del SE
    if (center[0] < 0 or center[0] >= SE.shape[0]) or (center[1] < 0 or center[1] >= SE.shape[1]):
        raise ValueError("El centro del elemento estructurante se sale del SE")

    outImage = np.zeros_like(inImage)

    #Calcular el padding basado en el centro del SE
    pad_y = center[0]
    pad_x = center[1]

    #Añadir padding de tipo constante con 0
    paddedImage = np.pad(inImage, ((pad_y, SE.shape[0] - center[0] - 1), (pad_x, SE.shape[1] - center[1] - 1)), mode='constant', constant_values=0)

    #Erosion
    for i in range(outImage.shape[0]):
        for j in range(outImage.shape[1]):

            region = paddedImage[i:i + SE.shape[0], j:j + SE.shape[1]]

            if np.all(region[SE == 1] == 1):    #Si todos son 1, entonces es 1 sino es un 0
                outImage[i, j] = 1

    return outImage


'''
outImage = dilate (inImage, SE, center=[])
    inImage, outImage: ...
    SE: Matriz PxQ de zeros y unos definiendo el elemento estructurante.
    center: Vector 1x2 con las coordenadas del centro de SE. Se asume que el [0 0] es
        la esquina superior izquierda. Si es un vector vacío (valor por defecto), el centro
        se calcula como (⌊P/2⌋ + 1, ⌊Q/2⌋ + 1).
'''
'''
def dilate(inImage, SE, center=DEFAULT_CENTER):

    #Verificar que el SE tiene dimensiones validas
    if SE.shape[0] < 1 or SE.shape[1] < 1:
        raise ValueError("El SE debe tener al menos dimension 1x1")
    
    #Calcular centro si no se especifica uno
    if not center:
        center = [SE.shape[0] // 2, SE.shape[1] // 2]  # Centro geométrico por defecto

    #Verificar que el centro no se sale del SE
    if (center[0] < 0 or center[0] >= SE.shape[0]) or (center[1] < 0 or center[1] >= SE.shape[1]):
        raise ValueError("El centro del elemento estructurante se sale del SE")

    outImage = np.zeros_like(inImage)

    #Calcular el padding basado en el centro del SE
    pad_y = center[0]
    pad_x = center[1]

    #Añadir padding de tipo constante con 0
    paddedImage = np.pad(inImage, ((pad_y, SE.shape[0] - center[0] - 1), (pad_x, SE.shape[1] - center[1] - 1)), mode='constant', constant_values=0)

    #Dilatación
    for i in range(outImage.shape[0]):
        for j in range(outImage.shape[1]):

            region = paddedImage[i:i + SE.shape[0], j:j + SE.shape[1]]

            if np.any(region[SE == 1] == 1):    #Si alguno es 1, entonces es 1 sino 0
                outImage[i, j] = 1

    return outImage
'''


def dilate(inImage, SE, center=DEFAULT_CENTER):

    #Verificar que el SE tiene dimensiones validas
    if SE.shape[0] < 1 or SE.shape[1] < 1:
        raise ValueError("El SE debe tener al menos dimension 1x1")
    
    #Calcular centro si no se especifica uno
    if not center:
        center = [SE.shape[0] // 2, SE.shape[1] // 2]  # Centro geométrico por defecto

    #Verificar que el centro no se sale del SE
    if (center[0] < 0 or center[0] >= SE.shape[0]) or (center[1] < 0 or center[1] >= SE.shape[1]):
        raise ValueError("El centro del elemento estructurante se sale del SE")

    

    #Calcular el padding basado en el centro del SE
    pad_y = center[0]
    pad_x = center[1]

    #Añadir padding de tipo constante con 0
    paddedImage = np.pad(inImage, ((pad_y, SE.shape[0] - center[0] - 1), (pad_x, SE.shape[1] - center[1] - 1)), mode='constant', constant_values=0)
    outImage = np.zeros_like(paddedImage)

    #Dilatación
    for i in range(paddedImage.shape[0]):
        for j in range(paddedImage.shape[1]):

            i_min = i - pad_y
            i_max = i + SE.shape[0] - center[0]
            j_min = j - pad_x
            j_max = j + SE.shape[1] - center[1]

            if(paddedImage[i,j]) == 1:
                outImage[i_min:i_max,j_min:j_max] = SE | outImage[i_min:i_max,j_min:j_max]
    
    #outImage = outImage | paddedImage #Creo que esta mal, lo que tengo que hacer en cada paso del bucle
    #es hacer la union de la region con el SE y lo que ya habia antes en outImage

    return outImage[pad_y : paddedImage.shape[0] - (SE.shape[0] - center[0] - 1), pad_x : paddedImage.shape[1] - (SE.shape[1] - center[1] - 1)]



'''
outImage = opening (inImage, SE, center=[])
    inImage, outImage: ...
    SE: Matriz PxQ de zeros y unos definiendo el elemento estructurante.
    center: Vector 1x2 con las coordenadas del centro de SE. Se asume que el [0 0] es
        la esquina superior izquierda. Si es un vector vacío (valor por defecto), el centro
        se calcula como (⌊P/2⌋ + 1, ⌊Q/2⌋ + 1).
'''
def opening(inImage, SE, center=DEFAULT_CENTER):
    eroded = erode(inImage, SE, center)
    outImage = dilate(eroded, SE, center)
    return outImage


'''
outImage = closing (inImage, SE, center=[])
    inImage, outImage: ...
    SE: Matriz PxQ de zeros y unos definiendo el elemento estructurante.
    center: Vector 1x2 con las coordenadas del centro de SE. Se asume que el [0 0] es
        la esquina superior izquierda. Si es un vector vacío (valor por defecto), el centro
        se calcula como (⌊P/2⌋ + 1, ⌊Q/2⌋ + 1).
'''
def closing(inImage, SE, center=DEFAULT_CENTER):
    dilated = dilate(inImage, SE, center)
    outImage = erode(dilated, SE, center)
    return outImage


'''
Implementar el algoritmo de llenado morfológico de regiones de una imagen, dado un
elemento estructurante de conectivdad, y una lista de puntos semilla

outImage = fill (inImage, seeds, SE=[], center=[])
    inImage, outImage, center: ...
    seeds: Matriz Nx2 con N coordenadas (fila,columna) de los puntos semilla.
    SE: Matriz PxQ de zeros y unos definiendo el elemento estructurante de conectividad.
        Si es un vector vacío se asume conectividad 4 (cruz 3 x 3).
'''
def fill(inImage, seeds, SE=DEFAULT_SE, center=DEFAULT_CENTER):
    
    outImage = np.zeros_like(inImage, dtype=np.uint8)

    #Añadir semillas a la imagen vacia
    for seed in seeds:
        if seed[0] < 0 or seed[0] >= inImage.shape[0] or seed[1] < 0 or seed[1] >= inImage.shape[1]:
            raise ValueError(f"Semilla {seed} está fuera de los límites de la imagen.")
        outImage[seed[0], seed[1]] = 1


    #Calcular centro si no se especifica uno
    if not center:
        center = [SE.shape[0] // 2, SE.shape[1] // 2]

    #Complementario de inImage
    inImageX = (inImage).astype(np.uint8) 
    Ac = 1 - inImageX

    
    while True:

        #Dilatacion
        prev_outImage = outImage.copy()
        outImage = dilate(outImage, SE, center)
        
        #Interseccion con el complemetario de inImage
        outImage = outImage & Ac

        #Para de iterar si no hay cambios
        if np.array_equal(outImage, prev_outImage):
            break

    #Unir resultado con imagen original 
    outImage = outImage | inImageX
    outImage = outImage.astype(inImage.dtype) #Volver al tipo original , float64

    return outImage
