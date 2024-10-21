import numpy as np

INPUT_IMAGES = 'input_images/'
OUTPUT_IMAGES = 'output_images/'

DEFAULT_INRANGE = []
DEFAULT_OUTRANGE = [0, 1]

DEFAULT_NBINS = 256

DEFAULT_CENTER = []
DEFAULT_SE = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)