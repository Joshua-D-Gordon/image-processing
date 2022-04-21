import math
import numpy as np
import cv2


def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """
    # cheecks wich one is bigger
    if len(in_signal) < len(k_size):
        a = k_size
        b = in_signal
    else:
        a = in_signal
        b = k_size
    #pads signal with 0's with length of b from each side so that when we "slide" over signal A with kernel B we have enough numbers for the whole "slide"
    a = np.pad(a, (len(b) - 1, len(b) - 1), 'constant', constant_values=(0, 0))
    convarr = np.zeros(len(a) - len(b) + 1)

    for i in range(len(a) - len(b) + 1):  # CHANGE TO MORE STEPS AND UNDERSTAND BETTER
        convarr[i] = (a[i:i + len(b)]*b).sum()

    return convarr


def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """
    #checks to see the bigger 2D array and asigns it to a and the smaller one to b
    x_img, y_img = in_image.shape
    x_kernel, y_kernel = kernel.shape
    if x_img * y_img >= x_kernel * y_kernel:
        a = in_image
        b = kernel
    else:
        a = kernel
        b = in_image

    # flips kernel for convolution;
    b = np.flip(b)

    #pads a with a reflection of the image at edges. we learnt this is the most usfull padding to use
    a_padded = np.pad(a, (b.shape[0] // 2, b.shape[1] // 2), 'edge')

    conv2Darr = np.zeros((a.shape[0], a.shape[1]))

    for i in range(conv2Darr.shape[0]):
        for j in range(conv2Darr.shape[1]):
            conv2Darr[i, j] = (a_padded[i:i + b.shape[0], j:j + b.shape[1]]*b).sum()

    return conv2Darr


def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """

    #set our X and Y kernels for our convoloution on image
    axis_x = np.array([[0, 0, 0],
                      [-1, 0, 1],
                      [0, 0, 0]]
                      )
    axis_y = np.array([[0, -1, 0],
                      [0, 0, 0],
                      [0, 1, 0]])
    #convolve on image and save aave each axis's differance
    x_output = conv2D(in_image, axis_x)
    y_output = conv2D(in_image, axis_y)

    #compute angle of magnitude
    angle_directions = np.arctan2(x_output, y_output)

    #compute magnitude
    magnitude = np.sqrt(np.square(x_output) + np.square(y_output))

    #return statment
    return angle_directions, magnitude



def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    #make sure k_size is an odd number
    if k_size % 2 == 0:
        k_size += 1

    #a good value for gaussian sigma as recomened by stackflow
    sigma = 0.3*((k_size - 1)*0.5 - 1) + 0.8

    # creating gaussian kernel
    gaussian_kernel = np.zeros((k_size, k_size))

    for i in range(k_size):
        for j in range(k_size):
            x = i - k_size // 2
            y = j - k_size // 2

            gaussian_kernel[i, j] = np.exp(-(x**2 + y**2)/(2 * sigma**2)) / (2 * np.pi * sigma ** 2)

    return conv2D(in_image, gaussian_kernel)



def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    #makes sure k size is a ood number
    if k_size%2 == 0:
        k_size += 1

    sigma = 0.3*((k_size - 1)*0.5 - 1) + 0.8
    kernel = cv2.getGaussianKernel(k_size, sigma)

    return cv2.filter2D(in_image, -1, kernel, borderType=cv2.BORDER_REPLICATE)


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """
    # create laplacian kernel for filter with image

    # [[0, 0, 0],[1, -2, 1],[0, 0, 0,]] + [[0, 1, 0],[0, -2, 0],[0, 1, 0]] =

    laplacian_k = np.array([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]])

    #convolution on smoothed image with laplacian kernel
    out_img = conv2D(img, laplacian_k)

    #2D array of zeros for output img - black canvas
    arr2D = np.zeros(out_img.shape)

    for x in range(out_img.shape[0] - laplacian_k.shape[0] - 1):
        for y in range(out_img.shape[1] - laplacian_k.shape[1] - 1):
            if out_img[x][y] == 0: # zero corrsing
                if(out_img[x-1][y] < 0 and out_img[x+1][y] > 0): # the value below on axis x is below zero and the value after above zero
                    arr2D[x][y] = 255 ## make white as there is a zero crossing event

                if(out_img[x-1][y] > 0 and out_img[x+1][y] < 0): # the value below on axis x is above zero and the value after below zero
                    arr2D[x][y] = 255 ## make white as there is a zero crossing event

                if(out_img[x][y-1] > 0 and out_img[x][y+1] < 0): # the value below on axis y is above zero and the value after below zero
                    arr2D[x][y] = 255 ## make white as there is a zero crossing event

                if(out_img[x][y-1] < 0 and out_img[x][y+1] > 0): # the value below on axis y is below zero and the value after above zero
                    arr2D[x][y] = 255 ## make white as there is a zero crossing event

            if out_img[x][y] < 0:
                if (out_img[x][y - 1] > 0) or (out_img[x][y + 1] > 0) or (out_img[x - 1][y] > 0) or (out_img[x + 1][y] > 0):
                    arr2D[x][y] = 255
    return arr2D


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """
    # smooth image first as laplacian is sensitive to noise as metioned in lecture although there will be less zero crossings
    img_smoothed = cv2.GaussianBlur(img, (9, 9), 0)

    out_img = edgeDetectionZeroCrossingSimple(img_smoothed)

    return out_img


def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    To find Edges you can Use OpenCV function: cv2.Canny
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
                [(x,y,radius),(x,y,radius),...]
    """
    img_blured_start = cv2.GaussianBlur(img, (3, 3), 0)
    img_edges_start = cv2.Canny(img_blured_start.astype(np.uint8), 50, 100)
    #blur image so that there are less fasley circels and compound these images together - enhances edges
    for i in range(5,21,2):
        img_blured = cv2.GaussianBlur(img, (i, i), 0)
        img_edges = cv2.Canny(img_blured.astype(np.uint8), 50, 100)
        img_edges_start += img_edges

    img = img_edges_start
    rows = img.shape[0]
    cols = img.shape[1]

    edges = []
    pts = []
    results = []
    circels = {}

    threshold = 0.4
    steps = 100

    for r in range(min_radius, max_radius + 1, 1):
        for s in range(steps):
            a = 2 * math.pi*s / steps
            x = int(r*np.cos(a))
            y = int(r*np.sin(a))
            pts.append((x, y, r))

    for i in range(rows):
        for j in range(cols):
            if img[i, j] == 255:
                edges.append((i, j))

    for e1, e2 in edges:
        for d1, d2, r in pts:
            a = e2 - d2
            b = e1 - d1
            s = circels.get((a, b, r))
            if s is None:
                s = 0
            circels[(a, b, r)] = s + 1

    sorted_circels = sorted(circels.items(), key=lambda i: -i[1])
    for circle, s in sorted_circels:
        x, y, r = circle
        if s / steps >= threshold and all((x -xc)*2 + (y - yc)*2 > rc ** 2 for xc, yc, rc in results):
            print(s/ steps, x, y, r)
            results.append((x, y, r))

    return results

def gaussian(n, sig):
    return (1.0/(2*np.pi*(sig**2)))*np.exp(-(n**2)/(2*(sig**2)))


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """
    #img to be returned
    image = np.zeros(in_image.shape)

    for row in range(len(image)):
        for col in range(len(image[0])):
            wp_total = 0
            filtered_image = 0
            for k in range(k_size):
                for l in range(k_size):
                    n_x = row - (k_size / 2 - k)
                    n_y = col - (k_size / 2 - l)
                    if n_x >= len(image):
                        n_x -= len(image)
                    if n_y >= len(image[0]):
                        n_y -= len(image[0])
                    gi = gaussian(image[int(n_x)][int(n_y)] - image[row][col], sigma_color)
                    gs = gaussian(np.sqrt(np.abs((n_x - row)**2-(n_y-col)**2)), sigma_space)
                    wp = gi * gs
                    filtered_image = (filtered_image) + (image[int(n_x)][int(n_y)] * wp)
                    wp_total = wp_total + wp
            filtered_image = filtered_image // wp_total
            image[row][col] = int(np.round(filtered_image))
    return image
