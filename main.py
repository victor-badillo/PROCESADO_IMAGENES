import numpy as np
import cv2
import os
from src import adjustIntensity, equalizeIntensity, filterImage, gaussKernel1D, gaussianFilter, medianFilter,erode, dilate, opening, closing, fill, gradientImage, LoG, edgeCanny
from utilidades import load_image, save_image_int, visualize_image, plot_histogram, histogram

if __name__ == "__main__":
    
    inputImage = load_image('circles1.png')

    outImage = edgeCanny(inputImage, 0.8, 0.1, 0.3)

    visualize_image('circle', outImage)
    save_image_int('circle.png', outImage)