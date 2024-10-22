import numpy as np
import cv2
from src import adjustIntensity, equalizeIntensity, filterImage, gaussKernel1D, gaussianFilter, medianFilter,erode, dilate, opening, closing, fill, gradientImage, LoG, edgeCanny
from utilidades import load_image, guardar_imagen_int, guardar_imagen_float, visualizar_imagen_int, visualizar_imagen_float
import matplotlib as plt

if __name__ == "__main__":
    
    inputImage = load_image('image2.png')

    outImage = gaussianFilter(inputImage, 0.8)
    outImage_adjust = adjustIntensity(outImage)
    guardar_imagen_int('mia.png', outImage_adjust)

    cv2Image = cv2.filter2D(inputImage, -1, gaussKernel1D(0.8))
    guardar_imagen_int('cv2.png', cv2Image)

    