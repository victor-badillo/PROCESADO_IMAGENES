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




#Expandir regiones blancas, puede conectar, aumentar tamaño, preparar para otras operaciones
'''
def dilate(inImage, SE, center=DEFAULT_CENTER):
    # Determinar el centro del elemento estructurante
    if not center:
        center = [SE.shape[0] // 2, SE.shape[1] // 2]


    # Verificar que el center sea adecuado para el tamaño del elemento estructurante
    if (center[0] < 0 or center[0] >= SE.shape[0]) or (center[1] < 0 or center[1] >= SE.shape[1]):
        raise ValueError("El valor de 'center' debe estar dentro de los límites del elemento estructurante.")

    # Crear la imagen de salida inicializada a ceros
    outImage = np.zeros_like(inImage)

    # Añadir padding a la imagen de entrada
    pad_y, pad_x = center
    #paddedImage = np.pad(inImage, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant', constant_values=0)
    paddedImage = np.pad(inImage, ((pad_y, pad_y), (pad_x, pad_x)), mode='reflect')

    # Aplicar dilatación
    for i in range(outImage.shape[0]):
        for j in range(outImage.shape[1]):
            # Extraer la región de interés
            region = paddedImage[i:i + SE.shape[0], j:j + SE.shape[1]]
            # Aplicar la operación de dilatación
            #Si alguno de la region es blanco, el del centro es blanco
            if np.any(region[SE == 1] == 1):
                outImage[i, j] = 1

    return outImage

'''

'''
outImage = dilate (inImage, SE, center=[])
    inImage, outImage: ...
    SE: Matriz PxQ de zeros y unos definiendo el elemento estructurante.
    center: Vector 1x2 con las coordenadas del centro de SE. Se asume que el [0 0] es
        la esquina superior izquierda. Si es un vector vacío (valor por defecto), el centro
        se calcula como (⌊P/2⌋ + 1, ⌊Q/2⌋ + 1).
'''
def dilate(inImage, SE, center=DEFAULT_CENTER):
    # Determinar el centro del elemento estructurante
    if not center:
        center = [SE.shape[0] // 2, SE.shape[1] // 2]  # Centro geométrico por defecto

    # Verificar que el 'center' esté dentro de los límites del elemento estructurante
    if (center[0] < 0 or center[0] >= SE.shape[0]) or (center[1] < 0 or center[1] >= SE.shape[1]):
        raise ValueError("El valor de 'center' debe estar dentro de los límites del elemento estructurante.")

    # Crear la imagen de salida inicializada a ceros
    outImage = np.zeros_like(inImage)

    # Calcular el padding basado en el centro del EE
    pad_y = center[0]  # Padding superior
    pad_x = center[1]  # Padding izquierdo

    # Añadir padding a la imagen de entrada
    paddedImage = np.pad(inImage, ((pad_y, SE.shape[0] - center[0] - 1), (pad_x, SE.shape[1] - center[1] - 1)), mode='constant', constant_values=0)
    #paddedImage = np.pad(inImage, ((pad_y, SE.shape[0] - center[0] - 1), (pad_x, SE.shape[1] - center[1] - 1)), mode='reflect')

    # Aplicar dilatación
    for i in range(outImage.shape[0]):
        for j in range(outImage.shape[1]):
            # Extraer la región de interés considerando el tamaño del SE
            region = paddedImage[i:i + SE.shape[0], j:j + SE.shape[1]]

            # Aplicar la operación de dilatación: si algún píxel de la región es blanco, se establece el píxel de salida a blanco
            if np.any(region[SE == 1] == 1):
                outImage[i, j] = 1

    return outImage
    
#Eliminar objetis pequeños o ruido , se preservan froma y tamaño de los objeto mas grandes
#Primero erosion para eliminar objetos pequeños y ruido y luego dilatacion para devolver al tamaño normal
'''
outImage = opening (inImage, SE, center=[])
    inImage, outImage: ...
    SE: Matriz PxQ de zeros y unos definiendo el elemento estructurante.
    center: Vector 1x2 con las coordenadas del centro de SE. Se asume que el [0 0] es
        la esquina superior izquierda. Si es un vector vacío (valor por defecto), el centro
        se calcula como (⌊P/2⌋ + 1, ⌊Q/2⌋ + 1).
'''
def opening(inImage, SE, center=DEFAULT_CENTER):
    # Apertura: erosión seguida de dilatación
    eroded = erode(inImage, SE, center)
    outImage = dilate(eroded, SE, center)
    return outImage



#Cerrar pequeños huevos y espacios dentro de los objetos
#Mejora conectividad y forma de los objetos
'''
outImage = closing (inImage, SE, center=[])
    inImage, outImage: ...
    SE: Matriz PxQ de zeros y unos definiendo el elemento estructurante.
    center: Vector 1x2 con las coordenadas del centro de SE. Se asume que el [0 0] es
        la esquina superior izquierda. Si es un vector vacío (valor por defecto), el centro
        se calcula como (⌊P/2⌋ + 1, ⌊Q/2⌋ + 1).
'''
def closing(inImage, SE, center=DEFAULT_CENTER):
    # Cierre: dilatación seguida de erosión
    dilated = dilate(inImage, SE, center)
    outImage = erode(dilated, SE, center)
    return outImage

# Rellenar regiones
# Usar visualiza_int
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
    
    # Crear imagen binaria inicial (X0) con los puntos semilla
    outImage = np.zeros_like(inImage, dtype=np.uint8)
    for seed in seeds:
        outImage[seed[0], seed[1]] = 1  # Colocar las semillas en X0


    # Si no se proporciona el centro, se coloca en el centro del SE
    if not center:
        center = [SE.shape[0] // 2, SE.shape[1] // 2]

    # Invertir la imagen de entrada para obtener el fondo (A^c)
    
    inImageX = (inImage).astype(np.uint8) 
    Ac = 1- inImageX

    
    while True:
        # Aplicar la dilatación
        prev_outImage = outImage.copy()
        #outImage = cv2.dilate(outImage, SE, anchor=center)
        outImage = dilate(outImage, SE, center)
        
        # Intersección con el fondo (A^c)
        outImage = outImage & Ac

        # Condición de terminación: si no hay cambios, detener
        if np.array_equal(outImage, prev_outImage):
            break

    # Unir la región final con la imagen original (Xk U A)    
    outImage = outImage | inImageX
    outImage = outImage.astype(inImage.dtype) #Volver al tipo original , float64

    return outImage
