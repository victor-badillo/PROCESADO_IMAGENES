import numpy as np
import cv2
import os
from src import adjustIntensity, equalizeIntensity, filterImage, gaussKernel1D, gaussianFilter, medianFilter,erode, dilate, opening, closing, fill, gradientImage, LoG, edgeCanny
from utilidades import load_image, save_image, visualize_image, plot_histogram, histogram

if __name__ == "__main__":
    
    inputImage_eq = load_image('eq0.png')
    outImage_eq = equalizeIntensity(inputImage_eq)
    outImage_adj = adjustIntensity(inputImage_eq)
    save_image('eq0_ecualizada.png', outImage_eq)
    save_image('eq0_ajustada.png', outImage_adj)

    inputImage_grid = load_image('grid.png')
    outImage_grid_3 = medianFilter(inputImage_grid, 3)
    outImage_grid_5 = medianFilter(inputImage_grid, 5)
    outImage_grid_7 = medianFilter(inputImage_grid, 7)
    save_image('grid_mediana_3.png', outImage_grid_3)
    save_image('grid_mediana_5.png', outImage_grid_5)
    save_image('grid_mediana_7.png', outImage_grid_7)

    inputImage_fill = load_image('image0.png')
    seeds_1 = [(0,0)]
    seeds_2 = [(25,25)]
    SE_4vecindad = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]], dtype=np.uint8)
    SE_8vecindad = np.array([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]], dtype=np.uint8)

    outImage_fill_1_four = fill(inputImage_fill,seeds_1, SE_4vecindad)
    outImage_fill_1_eight = fill(inputImage_fill, seeds_1, SE_8vecindad)
    outImage_fill_2_four = fill(inputImage_fill, seeds_2, SE_4vecindad)
    outImage_fill_2_eight = fill(inputImage_fill, seeds_2, SE_8vecindad)
    save_image('fill_1_four.png', outImage_fill_1_four)
    save_image('fill_1_eight.png', outImage_fill_1_eight)
    save_image('fill_2_four.png', outImage_fill_2_four)
    save_image('fill_2_eight.png', outImage_fill_2_eight)
    
    inputImage_canny = load_image('image5.png')
    outImage_low = edgeCanny(inputImage_canny, 3, 0.05, 0.2)
    outImage_high = edgeCanny(inputImage_canny, 3, 0.75, 0.85)
    outImage_diff = edgeCanny(inputImage_canny, 3, 0.3, 0.7)
    save_image('outImage_low.png', outImage_low)
    save_image('outImage_high.png', outImage_high)
    save_image('outImage_diff.png', outImage_diff)

    inputImage_log = load_image('image5.png')
    outImage_log = LoG(inputImage_log,2)
    save_image('log.png', outImage_log)

