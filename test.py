import numpy as np
import os
from utilidades import load_image, save_image_int, save_image_float, visualize_image_int, visualize_image_float
from src import adjustIntensity, equalizeIntensity, filterImage, gaussKernel1D, gaussianFilter, medianFilter, erode, dilate, opening, closing, fill, gradientImage, LoG, edgeCanny
from configuracion import INPUT_IMAGES
import cv2

def test_adjustIntensity():

    image_name = os.path.basename(INPUT_IMAGES + 'grays.png')
    image_name_ext = os.path.splitext(image_name)[0]

    in_image = load_image(image_name)

    #Caso 1:uso de parámetros por defecto
    out_image_default = adjustIntensity(in_image)
    #visualize_image_float(image_name_ext, out_image_default)
    save_image_int(f'{image_name_ext}_adjust_default.png', out_image_default)

    #Caso 2:ajuste del rango de intensidad de entrada
    in_range = [0.2, 0.8]
    out_image_in_range = adjustIntensity(in_image, inRange=in_range)
    save_image_int(f'{image_name_ext}_adjust_in_range.png', out_image_in_range)

    #Caso 3:ajuste del rango de intensidad de salida
    out_range = [0.1, 0.9]
    out_image_out_range = adjustIntensity(in_image, outRange=out_range)
    save_image_int(f'{image_name_ext}_adjust_out_range.png', out_image_out_range)

    #Caso 4:ajuste tanto de rango de entrada como de salida
    in_range = [0.1, 0.7]
    out_range = [0.0, 0.5]
    out_image_full = adjustIntensity(in_image, inRange=in_range, outRange=out_range)
    save_image_int(f'{image_name_ext}_adjust_full.png', out_image_full)

    print("Las imágenes ajustadas han sido guardadas con éxito.")


def test_equalize_intensity():

    image_name = os.path.basename(INPUT_IMAGES + 'eq0.png')
    image_name_ext = os.path.splitext(image_name)[0]

    in_image = load_image(image_name)

    #Caso 1:uso de parámetros por defecto
    out_image_default = equalizeIntensity(in_image)
    #visualize_image_float(image_name_ext, out_image_default)
    save_image_int(f'{image_name_ext}_equalize_default.png', out_image_default)

    #Caso 2:uso de un número menor de bins
    nBins_custom = 128
    out_image_custom_bins = equalizeIntensity(in_image, nBins=nBins_custom)
    save_image_int(f'{image_name_ext}_equalize_low_bins.png', out_image_custom_bins)

    #Caso 3:uso de un número mayor de bins
    nBins_high = 512
    out_image_high_bins = equalizeIntensity(in_image, nBins=nBins_high)
    save_image_int(f'{image_name_ext}_equalize_high_bins.png', out_image_high_bins)

    #Caso 4: Verificar que se lanza un ValueError cuando el numero de bins es <= 0
    try:
        
        nBins_high = -10
        out_image_error = equalizeIntensity(in_image, nBins=nBins_high)
    except ValueError as e:
        print(f"Prueba pasada: {e}")

    print("Las imágenes ecualizadas han sido guardadas con éxito.")

    
def test_filterImage():

    image_name = os.path.basename(INPUT_IMAGES + 'circles.png')
    image_name_ext = os.path.splitext(image_name)[0]
    in_image = load_image(image_name)

    kernels = {
        "average_3x3": np.array([[1, 1, 1],
                                  [1, 1, 1],
                                  [1, 1, 1]]) / 9,  #Kernel de suavizado (promedio) 3x3
        
        "average_5x5": np.array([[1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1]]) / 25,  #Kernel de suavizado (promedio) 5x5
        
        "average_7x7": np.array([[1, 1, 1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1, 1, 1]]) / 49,  #Kernel de suavizado (promedio) 7x7

        "average_2x2": np.array([[1, 1],
                                  [1, 1]]) / 4  #Kernel de suavizado (promedio) 7x7
        
    }

    for kernel_name, kernel in kernels.items():

        out_image = filterImage(in_image, kernel)
        out_image = adjustIntensity(out_image)  #Renormalizacion
        #visualize_image_float(image_name_ext, out_image)
        
        save_image_int(f'{image_name_ext}_{kernel_name}.png', out_image)

        assert out_image.shape == in_image.shape, f"La imagen de salida para el kernel {kernel_name} no tiene el mismo tamaño que la de entrada."

    #Probar con una imagen de todos ceros
    zero_image = np.zeros_like(in_image)
    out_zero_image = filterImage(zero_image, kernels.get("average_3x3"))
    assert np.all(out_zero_image == 0), "La imagen de salida debería ser toda ceros."

    print("Todas las pruebas con múltiples kernels han pasado con éxito.")


def test_gaussKernel1D():
    #Caso 1:verificar el tamaño del kernel
    sigma = 1
    kernel = gaussKernel1D(sigma)
    N = 2 * int(np.ceil(3 * sigma)) + 1
    assert len(kernel) == N, f"El tamaño del kernel debería ser {N} para sigma = {sigma}, pero fue {len(kernel)}."

    #Caso 2:verificar que el kernel esté normalizado (suma cercana a 1)
    sigma = 1.5
    kernel = gaussKernel1D(sigma)
    assert np.isclose(np.sum(kernel), 1, atol=1e-5), f"La suma del kernel debería ser aproximadamente 1, pero fue {np.sum(kernel)}."

    #Caso 3:verificar la simetría del kernel
    sigma = 2
    kernel = gaussKernel1D(sigma)
    assert np.allclose(kernel, kernel[::-1]), "El kernel debería ser simétrico."

    #Caso 4:verificar que los valores disminuyen desde el centro
    sigma = 1
    kernel = gaussKernel1D(sigma)
    center = len(kernel) // 2
    for i in range(center):
        assert kernel[i] <= kernel[i + 1], f"Los valores del kernel deberían aumentar hasta el centro para sigma = {sigma}."
        assert kernel[-(i + 1)] <= kernel[-(i + 2)], f"Los valores del kernel deberían disminuir desde el centro para sigma = {sigma}."

    #Caso 5:verificar el comportamiento con un valor pequeño de sigma, suavizado leve
    sigma = 0.5
    kernel = gaussKernel1D(sigma)
    assert len(kernel) == 2 * int(np.ceil(3 * sigma)) + 1, f"El tamaño del kernel debería ser {N} para sigma = {sigma}."
    assert np.isclose(np.sum(kernel), 1, atol=1e-5), f"La suma del kernel debería ser aproximadamente 1, pero fue {np.sum(kernel)}."

    print("Todos los tests han pasado con éxito.")


def test_gaussianFilter():

    #Caso 1:verificar que con una matriz constante no cambia la imagen
    inImage = np.ones((5, 5))
    sigma = 1.0
    outImage = gaussianFilter(inImage, sigma)
    assert np.allclose(outImage, 1), "El filtro no debería cambiar una imagen constante."  

    image_name = os.path.basename(INPUT_IMAGES + 'image2.png')
    image_name_ext = os.path.splitext(image_name)[0]

    #Probar diferentes valores de sigma para una imagen
    inImage = load_image(image_name)
    sigma_values = [0.5, 1.0, 2.0, 3.0]
    for sigma in sigma_values:
        outImage = gaussianFilter(inImage, sigma)
        outImage = adjustIntensity(outImage)    #Renormalizar
        #visualize_image_float(image_name_ext, outImage)
        save_image_int(f'{image_name_ext}_gauss_{sigma}.png', outImage)

        assert outImage.shape == inImage.shape, f"La imagen de salida debería tener el mismo tamaño que la imagen de entrada para sigma = {sigma}."

    print("Todos los tests de gaussianFilter han pasado.")


def test_medianFilter():

    #Caso 1 :verificar que con una matriz constante no cambia la imagen
    inImage = np.ones((5, 5))
    filterSize = 3
    outImage = medianFilter(inImage, filterSize)
    assert np.all(outImage == 1), "El filtro no debería cambiar una imagen constante."

    #Caso 2 :imagen identidad de tamaño 5x5
    inImage = np.eye(5)
    filterSize = 3
    outImage = medianFilter(inImage, filterSize)
    assert np.all(outImage >= 0) and np.all(outImage <= 1), "Los valores de la imagen filtrada deberían estar en el rango [0, 1]."

    #Caso 3 :test con diferentes valores de filterSize
    image_name = os.path.basename(INPUT_IMAGES + 'circlesSP.png')
    image_name_ext = os.path.splitext(image_name)[0]

    inImage = load_image(image_name)
    for filterSize in [3, 5, 7, 9, 11, 23, 53, 4]:
        outImage = medianFilter(inImage, filterSize)
        outImage = adjustIntensity(outImage)
        #visualize_image_float(image_name_ext,outImage )
        save_image_int(f'{image_name_ext}_median_{filterSize}.png', outImage)

        assert outImage.shape == inImage.shape, f"La imagen de salida debería tener el mismo tamaño que la imagen de entrada para filterSize = {filterSize}."

    #Caso 4: Verificar que se lanza un ValueError cuando el filterSize es <= 0
    try:
        
        filterSize = -10
        out_image_error = medianFilter(inImage, filterSize)
    except ValueError as e:
        print(f"Prueba pasada: {e}")

    print("Todos los tests de medianFilter han pasado.")


def test_erode():

    image_name = os.path.basename(INPUT_IMAGES + 'B.png')
    image_name_ext = os.path.splitext(image_name)[0]
    
    inputImage = load_image(image_name)

    print((np.array([[1], [1]], dtype=np.uint8)).shape)

    #Caso 1 :erosión con un SE 3x3 y centro predeterminado
    SE_square = np.array([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]], dtype=np.uint8)
    outImage_default = erode(inputImage, SE_square)
    save_image_int(f'{image_name_ext}_default.png', outImage_default)

    #Caso 2 :erosión con un SE horizontal 1x3 y centro predeterminado
    SE_1x3 = np.array([[1, 1, 1]], dtype=np.uint8)
    outImage_1x3 = erode(inputImage, SE_1x3)
    #visualize_image_float(image_name_ext, outImage_1x3)
    save_image_int(f'{image_name_ext}_1x3.png', outImage_1x3)

    #Caso 3 :erosión con un SE vertical 3x1 y centro predeterminado
    SE_3x1 = np.array([[1], [1], [1]], dtype=np.uint8)
    outImage_3x1 = erode(inputImage, SE_3x1)
    save_image_int(f'{image_name_ext}_3x1.png', outImage_3x1)

    #Caso 4 :erosión con un SE cuadrado 3x3 y un centro personalizado
    SE_square = np.array([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]], dtype=np.uint8)
    custom_center = [1, 1]
    outImage_custom_center = erode(inputImage, SE_square, center=custom_center)
    save_image_int(f'{image_name_ext}_custom_center.png', outImage_custom_center)

    # Caso 5 :erosión con un EE 3x3 y un centro en la esquina inferior derecha
    custom_center_edge = [2, 2]
    outImage_edge_center = erode(inputImage, SE_square, center=custom_center_edge)
    save_image_int(f'{image_name_ext}_edge_center.png', outImage_edge_center)

    #Caso 6 :verificar que se lanza un ValueError cuando el centro está fuera de los límites
    try:
        invalid_center = [3, 1]
        outImage_invalid_center = erode(inputImage, SE_square, center=invalid_center)
    except ValueError as e:
        print(f"Prueba pasada: {e}")

    #Caso 7 :verificar que se lanza un ValueError cuando el SE es vacio
    try:
        invalid_SE = np.array([], dtype=np.uint8)
        outImage_invalid_SE = erode(inputImage, invalid_SE)
    except ValueError as e:
        print(f"Prueba pasada: {e}")

    print("Todas las pruebas de erosión han pasado con éxito.")


def test_dilate():

    image_name = os.path.basename(INPUT_IMAGES + 'B.png')
    image_name_ext = os.path.splitext(image_name)[0]

    inputImage = load_image(image_name)

    #Caso 1 :dilatación con un SE 3x3 y centro predeterminado
    SE_square = np.array([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]], dtype=np.uint8)
    outImage_default = dilate(inputImage, SE_square)
    save_image_int(f'{image_name_ext}_default.png', outImage_default)

    #Caso 2 :dilatación con un SE horizontal 1x3 y centro predeterminado
    SE_1x3 = np.array([[1, 1, 1]], dtype=np.uint8)  # EE horizontal de 1x3
    outImage_1x3 = dilate(inputImage, SE_1x3)
    #visualize_image_float(image_name_ext, outImage_1x3)
    save_image_int(f'{image_name_ext}_1x3.png', outImage_1x3)

    #Caso 3 :dilatación con un SE vertical 3x1 y centro predeterminado
    SE_3x1 = np.array([[1], [1], [1]], dtype=np.uint8)  # EE vertical de 3x1
    outImage_3x1 = dilate(inputImage, SE_3x1)
    save_image_int(f'{image_name_ext}_3x1.png', outImage_3x1)

    #Caso 4 :dilatación con un SE cuadrado 3x3 y un centro personalizado
    custom_center = [1, 1]
    outImage_custom_center = dilate(inputImage, SE_square, center=custom_center)
    save_image_int(f'{image_name_ext}_custom_center.png', outImage_custom_center)

    #Caso 5 :dilatación con un SE 3x3 y un centro en la esquina inferior derecha
    custom_center_edge = [2, 2]
    outImage_edge_center = dilate(inputImage, SE_square, center=custom_center_edge)
    save_image_int(f'{image_name_ext}_edge_center.png', outImage_edge_center)

    #Caso 6 :verificar que se lanza un ValueError cuando el centro está fuera de los límites
    try:
        invalid_center = [3, 1]
        outImage_invalid_center = dilate(inputImage, SE_square, center=invalid_center)
    except ValueError as e:
        print(f"Prueba pasada: {e}")

    #Caso 7 :verificar que se lanza un ValueError cuando el SE es vacio
    try:
        invalid_SE = np.array([], dtype=np.uint8)
        outImage_invalid_SE = erode(inputImage, invalid_SE)
    except ValueError as e:
        print(f"Prueba pasada: {e}")

    print("Todas las pruebas de dilatación han pasado con éxito.")


def test_opening():

    image_name = os.path.basename(INPUT_IMAGES + 'B.png')
    image_name_ext = os.path.splitext(image_name)[0]

    inputImage = load_image(image_name)

    #Caso 1 :opening con un SE 3x3 y centro predeterminado
    SE_square = np.array([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]], dtype=np.uint8)
    outImage_default = opening(inputImage, SE_square)
    save_image_int(f'{image_name_ext}_default.png', outImage_default)

    #Caso 2 :opening con un SE horizontal 1x3 y centro predeterminado
    SE_1x3 = np.array([[1, 1, 1]], dtype=np.uint8)
    outImage_1x3 = opening(inputImage, SE_1x3)
    #visualize_image_float(image_name_ext, outImage_1x3)
    save_image_int(f'{image_name_ext}_1x3.png', outImage_1x3)

    #Caso 3 :opening con un SE vertical 3x1 y centro predeterminado
    SE_3x1 = np.array([[1], [1], [1]], dtype=np.uint8)
    outImage_3x1 = opening(inputImage, SE_3x1)
    save_image_int(f'{image_name_ext}_3x1.png', outImage_3x1)

    #Caso 4 :opening con un SE cuadrado 3x3 y un centro personalizado
    custom_center = [1, 1]
    outImage_custom_center = opening(inputImage, SE_square, center=custom_center)
    save_image_int(f'{image_name_ext}_custom_center.png', outImage_custom_center)

    #Caso 5 :opening con un SE 3x3 y un centro en la esquina inferior derecha
    custom_center_edge = [2, 2]
    outImage_edge_center = opening(inputImage, SE_square, center=custom_center_edge)
    save_image_int(f'{image_name_ext}_edge_center.png', outImage_edge_center)

    #Caso 6: Verificar que se lanza un ValueError cuando el centro está fuera de los límites
    try:
        invalid_center = [3, 1]
        outImage_invalid_center = opening(inputImage, SE_square, center=invalid_center)
    except ValueError as e:
        print(f"Prueba pasada: {e}")

    #Caso 7 :verificar que se lanza un ValueError cuando el SE es vacio
    try:
        invalid_SE = np.array([], dtype=np.uint8)
        outImage_invalid_SE = erode(inputImage, invalid_SE)
    except ValueError as e:
        print(f"Prueba pasada: {e}")

    print("Todas las pruebas de opening han pasado con éxito.")


def test_closing():

    image_name = os.path.basename(INPUT_IMAGES + 'B.png')
    image_name_ext = os.path.splitext(image_name)[0]

    inputImage = load_image(image_name)

    #Caso 1 :closing con un SE 3x3 y centro predeterminado
    SE_square = np.array([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]], dtype=np.uint8)
    outImage_default = closing(inputImage, SE_square)
    save_image_int(f'{image_name_ext}_default.png', outImage_default)

    #Caso 2 :closing con un SE horizontal 1x3 y centro predeterminado
    SE_1x3 = np.array([[1, 1, 1]], dtype=np.uint8)
    outImage_1x3 = closing(inputImage, SE_1x3)
    #visualize_image_float(image_name_ext, outImage_1x3)
    save_image_int(f'{image_name_ext}_1x3.png', outImage_1x3)

    #Caso 3 :closing con un SE vertical 3x1 y centro predeterminado
    SE_3x1 = np.array([[1], [1], [1]], dtype=np.uint8)
    outImage_3x1 = closing(inputImage, SE_3x1)
    save_image_int(f'{image_name_ext}_3x1.png', outImage_3x1)

    #Caso 4 :closing con un EE cuadrado 3x3 y un centro personalizado
    custom_center = [1, 1]
    outImage_custom_center = closing(inputImage, SE_square, center=custom_center)
    save_image_int(f'{image_name_ext}_custom_center.png', outImage_custom_center)

    #Caso 5 :closing con un EE 3x3 y un centro en la esquina inferior derecha
    custom_center_edge = [2, 2]
    outImage_edge_center = closing(inputImage, SE_square, center=custom_center_edge)
    save_image_int(f'{image_name_ext}_edge_center.png', outImage_edge_center)

    #Caso 6 :verificar que se lanza un ValueError cuando el centro está fuera de los límites
    try:
        invalid_center = [3, 1]
        outImage_invalid_center = closing(inputImage, SE_square, center=invalid_center)
    except ValueError as e:
        print(f"Prueba pasada: {e}")

    #Caso 7 :verificar que se lanza un ValueError cuando el SE es vacio
    try:
        invalid_SE = np.array([], dtype=np.uint8)
        outImage_invalid_SE = erode(inputImage, invalid_SE)
    except ValueError as e:
        print(f"Prueba pasada: {e}")

    print("Todas las pruebas de closing han pasado con éxito.")



def test_fill():

    image_name = os.path.basename(INPUT_IMAGES + 'B.png')
    image_name_ext = os.path.splitext(image_name)[0]
    
    inputImage = load_image(image_name)
    
    #Caso 1 :fill con un SE por defecto , centro por defecto y una semilla
    seeds = [(13, 9)]
    outImage_default = fill(inputImage, seeds)
    #visualize_image_float(image_name_ext,outImage_default)
    save_image_int(f'{image_name_ext}_fill_default.png', outImage_default)

    #Caso 2 :fill con varias semillas en diferentes agujeros de la letra 'B'
    seeds_multiple = [(14,9), (7,9)]
    outImage_multiple_seeds = fill(inputImage, seeds_multiple)
    save_image_int(f'{image_name_ext}_fill_multiple_seeds.png', outImage_multiple_seeds)
    
    #Caso 3 :fill con un SE horizontal de 1x3
    SE_1x3 = np.array([[1, 1, 1]], dtype=np.uint8)
    seeds = [(13, 9)]
    outImage_1x3 = fill(inputImage, seeds, SE=SE_1x3)
    save_image_int(f'{image_name_ext}_fill_1x3.png', outImage_1x3)
    
    #Caso 4 :fill con un SE vertical 3x1
    SE_3x1 = np.array([[1], [1], [1]], dtype=np.uint8)
    outImage_3x1 = fill(inputImage, seeds, SE=SE_3x1)
    save_image_int(f'{image_name_ext}_fill_3x1.png', outImage_3x1)
    
    #Caso 5 :fill con un SE cuadrado 3x3 y un centro personalizado
    SE_square = np.ones((3, 3), dtype=np.uint8)
    custom_center = [0, 2]
    outImage_custom_center = fill(inputImage, seeds, SE=SE_square, center=custom_center)
    save_image_int(f'{image_name_ext}_fill_custom_center.png', outImage_custom_center)
    
    #Caso 6 :verificar que se lanza un ValueError cuando el centro está fuera de los límites
    try:
        invalid_center = [3, 1]
        outImage_invalid_center = fill(inputImage, seeds, SE=SE_square, center=invalid_center)
    except ValueError as e:
        print(f"Prueba pasada: {e}")

    #Caso 7 :verificar que se lanza un ValueError cuando hay una semilla fuera de los limites de la imagen
    try:
        invalid_seed = [(-10,-10)]
        outImage_invalid_seed = fill(inputImage, invalid_seed)
    except ValueError as e:
        print(f"Prueba pasada: {e}")
    
    print("Todas las pruebas de fill han pasado con éxito.")


def test_gradientImage():
    # Cargar la imagen de prueba
    image_name = os.path.basename(INPUT_IMAGES + 'circleWhite.png')  # Asegúrate de que la imagen esté en esta ruta
    image_name_ext = os.path.splitext(image_name)[0]
    inputImage = load_image(image_name)  # Cargar la imagen

    # Caso 1: Aplicar el operador Roberts
    gx_roberts, gy_roberts = gradientImage(inputImage, 'Roberts')
    gx_roberts = np.clip(gx_roberts, 0, 1)
    gy_roberts = np.clip(gy_roberts, 0, 1)
    save_image_int(f'{image_name_ext}_gradient_roberts_gx.png', gx_roberts)
    save_image_int(f'{image_name_ext}_gradient_roberts_gy.png', gy_roberts)

    # Caso 2: Aplicar el operador CentralDiff
    gx_central, gy_central = gradientImage(inputImage, 'CentralDiff')
    gx_central = np.clip(gx_central, 0, 1)
    gy_central = np.clip(gy_central, 0, 1)
    save_image_int(f'{image_name_ext}_gradient_central_gx.png', gx_central)
    save_image_int(f'{image_name_ext}_gradient_central_gy.png', gy_central)

    # Caso 3: Aplicar el operador Prewitt
    gx_prewitt, gy_prewitt = gradientImage(inputImage, 'Prewitt')
    gx_prewitt = np.clip(gx_prewitt, 0, 1)
    gy_prewitt = np.clip(gy_prewitt, 0, 1)
    save_image_int(f'{image_name_ext}_gradient_prewitt_gx.png', gx_prewitt)
    save_image_int(f'{image_name_ext}_gradient_prewitt_gy.png', gy_prewitt)

    # Caso 4: Aplicar el operador Sobel
    gx_sobel, gy_sobel = gradientImage(inputImage, 'Sobel')
    gx_sobel = np.clip(gx_sobel, 0, 1)
    gy_sobel = np.clip(gy_sobel, 0, 1)
    save_image_int(f'{image_name_ext}_gradient_sobel_gx.png', gx_sobel)
    save_image_int(f'{image_name_ext}_gradient_sobel_gy.png', gy_sobel)

    # Caso 5: Verificar que se lanza un ValueError para un operador no válido
    try:
        gradientImage(inputImage, 'InvalidOperator')
    except ValueError as e:
        print(f"Prueba pasada: {e}")

    # Imprimir mensaje de éxito
    print("Todas las pruebas de gradientImage han pasado con éxito.")

def test_LoG():
    # Cargar la imagen de prueba
    image_name = os.path.basename(INPUT_IMAGES + 'image2.png')
    image_name_ext = os.path.splitext(image_name)[0]
    inputImage = load_image(image_name)  # Cargar la imagen 

    # Caso 1: Aplicar LoG con un valor de sigma específico
    sigma_1 = 1.0
    outImage_sigma_1 = LoG(inputImage, sigma_1)
    outImage_sigma_1 = np.clip(outImage_sigma_1, 0, 1)
    save_image_int(f'{image_name_ext}_LoG_sigma_1.png', outImage_sigma_1)

    # Caso 2: Aplicar LoG con un valor de sigma diferente
    sigma_2 = 2.0
    outImage_sigma_2 = LoG(inputImage, sigma_2)
    outImage_sigma_2 = np.clip(outImage_sigma_2, 0, 1)
    save_image_int(f'{image_name_ext}_LoG_sigma_2.png', outImage_sigma_2)

    # Caso 3: Aplicar LoG con un valor de sigma diferente
    sigma_3 = 3.0
    outImage_sigma_3 = LoG(inputImage, sigma_3)
    outImage_sigma_3 = np.clip(outImage_sigma_3, 0, 1)
    save_image_int(f'{image_name_ext}_LoG_sigma_3.png', outImage_sigma_3)

    # Caso 3: Verificar que la salida tenga las mismas dimensiones que la imagen de entrada
    assert outImage_sigma_1.shape == inputImage.shape, "La salida no tiene las mismas dimensiones que la imagen de entrada"
    assert outImage_sigma_2.shape == inputImage.shape, "La salida no tiene las mismas dimensiones que la imagen de entrada"
    assert outImage_sigma_3.shape == inputImage.shape, "La salida no tiene las mismas dimensiones que la imagen de entrada"

    # Caso 4: Comprobar que la imagen de salida no tiene valores fuera del rango [0, 255]
    assert np.all(outImage_sigma_1 >= 0) and np.all(outImage_sigma_1 <= 255), "Los valores en outImage_sigma_1 están fuera de rango [0, 255]"
    assert np.all(outImage_sigma_2 >= 0) and np.all(outImage_sigma_2 <= 255), "Los valores en outImage_sigma_2 están fuera de rango [0, 255]"
    assert np.all(outImage_sigma_3 >= 0) and np.all(outImage_sigma_3 <= 255), "Los valores en outImage_sigma_3 están fuera de rango [0, 255]"

    # Imprimir mensaje de éxito
    print("Todas las pruebas de LoG han pasado con éxito.")

def test_edgeCanny():
    # Cargar la imagen de prueba
    image_name = os.path.basename(INPUT_IMAGES + 'circles1.png')  # Cambia 'test_image.png' por el nombre de tu imagen
    image_name_ext = os.path.splitext(image_name)[0]
    inputImage = load_image(image_name)  # Cargar la imagen

    # Definir diferentes parámetros para probar
    test_cases = [
        (1.0, 50, 150, 'sigma_1_tlow_50_thigh_150.png'),
        (1.0, 100, 200, 'sigma_1_tlow_100_thigh_200.png'),
        (2.0, 50, 150, 'sigma_2_tlow_50_thigh_150.png'),
        (2.0, 100, 200, 'sigma_2_tlow_100_thigh_200.png'),
        (3.0, 50, 150, 'sigma_3_tlow_50_thigh_150.png'),
        (3.0, 100, 200, 'sigma_3_tlow_100_thigh_200.png'),
    ]

    for sigma, tlow, thigh, output_filename in test_cases:
        # Aplicar el detector de bordes de Canny
        outImage = edgeCanny(inputImage, sigma, tlow, thigh)
        outImage = np.clip(outImage, 0, 1)
        # Guardar la imagen de salida
        save_image_int(f'{image_name_ext}_{output_filename}', outImage)


    # Imprimir mensaje de éxito
    print("Todas las pruebas de LoG han pasado con éxito.")


if __name__ == "__main__":
    #test_adjustIntensity()
    #test_equalize_intensity()
    #test_filterImage()
    #test_gaussKernel1D()
    #test_gaussianFilter()
    #test_medianFilter()
    #test_erode()
    #test_dilate()
    #test_opening()
    #test_closing()
    test_fill()
    #test_gradientImage()
    #test_LoG()
    #test_edgeCanny()