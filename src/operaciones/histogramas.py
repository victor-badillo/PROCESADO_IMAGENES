import numpy as np
from configuracion import DEFAULT_INRANGE, DEFAULT_OUTRANGE, DEFAULT_NBINS
from utilidades import histogram


def adjustIntensity(inImage, inRange=DEFAULT_INRANGE, outRange=DEFAULT_OUTRANGE):
    #Si no se especifica inRange utilizar el min y max de la imagen de entrada
    if not inRange:
        imin = np.min(inImage)
        imax = np.max(inImage)
    else:
        imin, imax = inRange
    
    omin, omax = outRange
    outImage = np.zeros_like(inImage)
    
    for i in range(inImage.shape[0]):
        for j in range(inImage.shape[1]):

            newValue = (inImage[i, j] - imin) * (omax - omin) / (imax - imin) + omin
            #Renormalización
            outImage[i, j] = np.clip(newValue, omin, omax) #Asegura que no se salga del rango especificado para la salida

    return outImage


def equalizeIntensity(inImage, nBins=DEFAULT_NBINS):

    #Calcular el histograma y los límites de los bins
    hist, bin_edges = histogram(inImage, bins=nBins, rango_min=0.0, rango_max=1.0)

    #Calcular la función de distribución acumulativa (CDF)
    cdf = hist.cumsum()  # Sumar el histograma acumulativo
    cdf_normalized = cdf / cdf[-1]  # Normalizar la CDF dividiendo entre el mayor valor

    #Mapeo de los valores de intensidad usando la CDF
    outImage = np.interp(inImage.flatten(), bin_edges[:-1], cdf_normalized)  #Usar los límites de los bins, el ultimo no representa un limite inferior
    outImage = outImage.reshape(inImage.shape)  #Devolver forma original a la imagen modificada

    #outImage = np.clip(outImage, 0, 1)  #Asegurar valores dentro del rango [0, 1]
    outImage = adjustIntensity(outImage)

    return outImage


