import numpy as np
import cv2
import os
from src import adjustIntensity, equalizeIntensity, filterImage, gaussKernel1D, gaussianFilter, medianFilter,erode, dilate, opening, closing, fill, gradientImage, LoG, edgeCanny
from utilidades import load_image, save_image, visualize_image, plot_histogram, histogram

if __name__ == "__main__":
    
    inputImage = load_image('circles.png')

    outImage = edgeCanny(inputImage, 0.8, 0.1, 0.3)

    visualize_image('circles', adjustIntensity(outImage))
    save_image('circles_salida.png', outImage)

