import numpy as np
import cv2
from src import adjustIntensity, equalizeIntensity, filterImage, gaussKernel1D, gaussianFilter, medianFilter,erode, dilate, opening, closing, fill, gradientImage, LoG, edgeCanny
from utilidades import load_image, save_image_int, save_image_float, visualize_image_int, visualize_image_float, plot_histogram, histogram
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    # inputImage = load_image('image2.png')

    # outImage = gaussianFilter(inputImage, 0.8)
    # outImage_adjust = adjustIntensity(outImage)
    # save_image_int('mia.png', outImage_adjust)

    # cv2Image = cv2.filter2D(inputImage, -1, gaussKernel1D(0.8))
    # save_image_int('cv2.png', cv2Image)

    # input_image = load_image('eq0.png')
    # hist, bin_edges = histogram(input_image, 256)
    # plot_histogram(hist, bin_edges)

    # output_image = equalizeIntensity(input_image)
    # hist, bin_edges = histogram(output_image, 256)
    # plot_histogram(hist, bin_edges)

    # save_image_int('dopaturbo.png', output_image)

    # output_image = cv2.equalizeHist((input_image * 255).astype(np.uint8))
    # hist, bin_edges = histogram(output_image, 256)
    # plot_histogram(hist, bin_edges)
    



    # visualize_image_float('epale', output_image)
    # save_image_float('dopanete.png', output_image)

    
    