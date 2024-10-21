import numpy as np
import cv2
from src import adjustIntensity, equalizeIntensity, filterImage, gaussKernel1D, gaussianFilter, medianFilter,erode, dilate, opening, closing, fill, gradientImage, LoG, edgeCanny
from utilidades import cargar_imagen, guardar_imagen_int, guardar_imagen_float, visualizar_imagen_int, visualizar_imagen_float


if __name__ == "__main__":
    
    inputImage = cargar_imagen('circlesSP.png')

    outImage = medianFilter(inputImage, 7)

    visualizar_imagen_int("klk", outImage)

    guardar_imagen_int('cv2.png', outImage)

    