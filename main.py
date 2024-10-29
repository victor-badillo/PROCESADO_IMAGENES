import numpy as np
import cv2
from src import adjustIntensity, equalizeIntensity, filterImage, gaussKernel1D, gaussianFilter, medianFilter,erode, dilate, opening, closing, fill, gradientImage, LoG, edgeCanny
from utilidades import load_image, save_image_int, save_image_float, visualize_image_int, visualize_image_float, plot_histogram, histogram
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    inputImage = load_image('circles.png')
   
    image = np.array([
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    ], dtype=np.uint8)

    SE_1x3 = np.array([[1, 1]], dtype=np.uint8)
    SE_1x3_cero = np.array([[1, 0, 1]], dtype=np.uint8)

    outImage = dilate(image, SE_1x3_cero, center=[0,1])

    print(outImage)