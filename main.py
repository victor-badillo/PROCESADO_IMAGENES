import numpy as np
import cv2
from src import adjustIntensity, equalizeIntensity, filterImage, gaussKernel1D, gaussianFilter, medianFilter,erode, dilate, opening, closing, fill, gradientImage, LoG, edgeCanny
from utilidades import load_image, save_image_int, save_image_float, visualize_image_int, visualize_image_float, plot_histogram, histogram
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    inputImage = load_image('lady.png')

    #Caso 1 : Roberts
    gx_roberts, gy_roberts = gradientImage(inputImage, 'Sobel')
    #gx_roberts = adjustIntensity(gx_roberts)
    #gy_roberts = adjustIntensity(gy_roberts)
    #gx_roberts = np.clip(gx_roberts, 0, 1)
    #gy_roberts = np.clip(gy_roberts, 0, 1)
    roberts_x = np.array([[1, 0],
                      [0, -1]], dtype=np.float32)

    roberts_y = np.array([[0, 1],
                        [-1, 0]], dtype=np.float32)

    # Aplicar el filtro de Roberts en X y Y
    gx = cv2.filter2D(inputImage, cv2.CV_64F, roberts_x)
    gy = cv2.filter2D(inputImage, cv2.CV_64F, roberts_y)
    save_image_int('gx.png', gx_roberts)
    save_image_int('gy.png', gy_roberts)