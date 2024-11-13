import numpy as np
import cv2
import os
from src import adjustIntensity, equalizeIntensity, filterImage, gaussKernel1D, gaussianFilter, medianFilter,erode, dilate, opening, closing, fill, gradientImage, LoG, edgeCanny
from utilidades import load_image, save_image, visualize_image, plot_histogram, histogram

if __name__ == "__main__":
    
    # inputImage = load_image('circles1.png')

    # outImage = edgeCanny(inputImage, 0.8, 0.1, 0.3)

    # visualize_image('circle', outImage)
    # save_image('circle.png', outImage)

    example_slides = np.array([
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float64)

    SE_1x3_cero = np.array([[1, 0, 1]], dtype=np.uint8)
    seeds = [(1,1)]
    outImage_slides_1x3 = fill(example_slides,seeds,  SE_1x3_cero)
    #outImage_slides_1x3 = dilate(example_slides, SE_1x3_cero)

    print(outImage_slides_1x3)
    print(outImage_slides_1x3.dtype)