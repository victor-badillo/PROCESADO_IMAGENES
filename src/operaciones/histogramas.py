import numpy as np
from configuracion import DEFAULT_INRANGE, DEFAULT_OUTRANGE, DEFAULT_NBINS
from utilidades import histogram

'''
def adjustIntensity(inImage, inRange=DEFAULT_INRANGE, outRange=DEFAULT_OUTRANGE):
    
    if not inRange: #Si no se especifica inRange utilizar el min y max de la imagen de entrada
        imin = np.min(inImage)
        imax = np.max(inImage)
    else:
        imin, imax = inRange
    
    omin, omax = outRange
    outImage = np.zeros_like(inImage)   #Matriz de zeros de mismas dimensiones que inImage
    
    for i in range(inImage.shape[0]):
        for j in range(inImage.shape[1]):

            newValue = (inImage[i, j] - imin) * (omax - omin) / (imax - imin) + omin
            #Renormalización
            outImage[i, j] = np.clip(newValue, omin, omax) #Asegurar que el nuevo valor este dentro del rango

    return outImage
'''

'''
Funcion para ajustar la intensidad de una imagen

outImage = adjustIntensity (inImage, inRange=[], outRange=[0 1])
    inImage: Matriz MxN con la imagen de entrada.
    outImage: Matriz MxN con la imagen de salida.
    inRange: Vector 1x2 con el rango de niveles de intensidad [imin, imax] de entrada.
        Si el vector está vacío (por defecto), el mínimo y máximo de la imagen de entrada
        se usan como imin e imax.
    outRange: Vector 1x2 con el rango de niveles de instensidad [omin, omax] de salida.
        El valor por defecto es [0 1].
'''
def adjustIntensity(inImage, inRange=DEFAULT_INRANGE, outRange=DEFAULT_OUTRANGE):
    
    if not inRange:  #Si no se especifica inRange utilizar el min y max de la imagen de entrada
        imin = np.min(inImage)
        imax = np.max(inImage)
    else:
        imin, imax = inRange
    
    omin, omax = outRange

    out_image = (omax - omin) * (inImage - imin) / (imax - imin) + omin

    # Asegurar que el nuevo valor esté dentro del rango
    out_image = np.clip(out_image, omin, omax)

    return out_image


'''
Funcion para ecualizar el histograma de una imagen

outImage = equalizeIntensity (inImage, nBins=256)
    inImage, outImage: ...
    nBins: Número de bins utilizados en el procesamiento. Se asume que el intervalo de
        entrada [0 1] se divide en nBins intervalos iguales para hacer el procesamiento,
        y que la imagen de salida vuelve a quedar en el intervalo [0 1]. Por defecto 256.
'''
def equalizeIntensity(inImage, nBins=DEFAULT_NBINS):

    if nBins <= 0:
        raise ValueError("El numero de bins debe ser mayor que 0")

    #Calcular el histograma y los límites de los bins
    hist, bin_edges = histogram(inImage, bins=nBins, minRange=0.0, maxRange=1.0)

    #Calcular la función de distribución acumulativa(CDF)
    cdf = hist.cumsum()  #Sumar el histograma acumulativo
    if cdf[-1] == 0:
        return np.zeros_like(inImage)  # Evitar división por cero
    cdf_normalized = cdf / cdf[-1]  #Normalizar

    #Interpolacion
    out_image = np.interp(inImage.flatten(), bin_edges[:-1], cdf_normalized)  #Usar los límites de los bins, el ultimo no , representa un limite inferior
    out_image = out_image.reshape(inImage.shape)  #Devolver forma original a la imagen modificada

    out_image = adjustIntensity(out_image) #Renormalizar

    return out_image


