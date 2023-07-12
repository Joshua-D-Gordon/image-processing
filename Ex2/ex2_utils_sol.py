import numpy as np
import cv2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 100


def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """

    if len(in_signal.shape) > 1:
        if in_signal.shape[1] > 1:
            raise ValueError("Input Signal is not a 1D array")
        else:
            in_signal = in_signal.reshape(in_signal.shape[0])
    inv_k = k_size[::-1].astype(np.float64)
    kernel_len = len(k_size)
    out_len = max(kernel_len, len(in_signal) + (kernel_len - 1))
    padding = kernel_len - 1
    padded_signal = np.pad(in_signal, padding, 'constant')
    out_signal = np.ones(out_len)
    for i in range(out_len):
        st = i
        end = i + kernel_len
        out_signal[i] = (padded_signal[st:end] * inv_k).sum()
    return out_signal


def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """

    h, w = in_image.shape[:2]
    k_size = np.array([x for x in kernel.shape])
    half = k_size // 2

    pad_img = np.pad(in_image.astype(np.float32),
                     ((k_size[0], k_size[0]), (k_size[1], k_size[1])),
                     'reflect')

    conv_img = np.zeros_like(in_image)

    for i in range(h):
        for j in range(w):
            x = j + half[1] + 1
            y = i + half[0] + 1
            conv_img[i, j] = (pad_img[y:y + k_size[0], x:x + k_size[1]] * kernel).sum()
    return conv_img


def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale image
    :return: (directions, magnitude)
    """

    kernel = np.array([[1, 0, -1]])
    x_drive = conv2D(in_image, kernel)
    y_drive = conv2D(in_image, kernel.T)
    ori = np.arctan2(y_drive, x_drive)
    mag = np.sqrt(x_drive ** 2 + y_drive ** 2)
    return ori, mag


def findZeroCrossing(lap_img: np.ndarray) -> np.ndarray:
    minLoG = cv2.morphologyEx(lap_img, cv2.MORPH_ERODE, np.ones((3, 3)))
    maxLoG = cv2.morphologyEx(lap_img, cv2.MORPH_DILATE, np.ones((3, 3)))
    zeroCross = np.logical_or(np.logical_and(minLoG < 0, lap_img > 0),
                              np.logical_and(maxLoG > 0, lap_img < 0))
    return zeroCross.astype(np.float)


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """

    my_laplacian = cv2.Laplacian(img.astype(np.float), -1)
    return findZeroCrossing(my_laplacian)


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """

    laplacian = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])
    my_gauss = createGaussianKernel(101)
    my_lap = cv2.filter2D(img.astype(np.float), -1, my_gauss)
    my_lap = cv2.filter2D(my_lap.astype(np.float), -1, laplacian)
    return findZeroCrossing(my_lap)


def createGaussianKernel(k_size: int):
    if k_size % 2 == 0:
        raise ValueError("Kernel size should be an odd number")

    k = np.array([1, 1], dtype=np.float64)
    iter_v = np.array([1, 1], dtype=np.float64)

    for i in range(2, k_size):
        k = conv1D(k, iter_v)
    k = k.reshape((len(k), 1))
    kernel = k.dot(k.T)
    kernel = kernel / kernel.sum()

    return kernel


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    kernel = createGaussianKernel(k_size)
    return conv2D(in_image, kernel)


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    kernel = cv2.getGaussianKernel(k_size, -1)
    kernel = kernel.dot(kernel.T)
    blurred_img = cv2.filter2D(in_image, -1, kernel)
    return blurred_img


def nms(xyr: np.ndarray, radius: int) -> list:
    """
    Performs Non Maximum Suppression in order to remove circles that are close
    to each other to get a "clean" output.
    :param xyr:
    :param radius:
    :return:
    """

    ret_xyr = []

    while len(xyr) > 0:
        # Choose most ranked circle (MRC)
        curr_arg = xyr[:, -1].argmax()
        curr = xyr[curr_arg, :]
        ret_xyr.append(curr)
        xyr = np.delete(xyr, curr_arg, axis=0)

        # Find MRC close neighbors
        dists = np.sqrt(np.square(xyr[:, :2] - curr[:2]).sum(axis=1)) < radius
        idx_to_delete = np.where(dists)

        # Delete MRCs neighbors
        xyr = np.delete(xyr, idx_to_delete, axis=0)
    return ret_xyr


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

    img = img.squeeze()
    if img.ndim > 2:
        raise ValueError("The image is not grayscale")
    h, w = img.shape
    max_radius = min((min(h, w) // 2), max_radius)

    # Get each pixels gradients direction
    i_y = cv2.Sobel(img, -1, 0, 1, ksize=3)
    i_x = cv2.Sobel(img, -1, 1, 0, ksize=3)
    ori = np.arctan2(i_y, i_x)

    # Get Edges using Canny Edge detector
    bw = cv2.Canny((img * 255).astype(np.uint8), 550, 100)
    radius_diff = max_radius - min_radius
    circle_hist = np.zeros((h, w, radius_diff))

    # Get the coordinates only for the edges
    ys, xs = np.where(bw)

    # Calculate the sin/cos for each edge pixel
    sins = np.sin(ori[ys, xs])
    coss = np.cos(ori[ys, xs])
    r_range = np.arange(min_radius, max_radius)

    for iy, ix, ss, cs in zip(ys, xs, sins, coss):
        grad_sin = (r_range * ss).astype(np.int)
        grad_cos = (r_range * cs).astype(np.int)

        xc_1 = ix + grad_cos
        yc_1 = iy + grad_sin

        xc_2 = ix - grad_cos
        yc_2 = iy - grad_sin

        # Check where are the centers that are in the image
        r_idx1 = np.logical_and(yc_1 > 0, xc_1 > 0)
        r_idx1 = np.logical_and(r_idx1, np.logical_and(yc_1 < h, xc_1 < w))

        # Check where are the centers that are in the image (Opposite direction)
        r_idx2 = np.logical_and(yc_2 > 0, xc_2 > 0)
        r_idx2 = np.logical_and(r_idx2, np.logical_and(yc_2 < h, xc_2 < w))

        # Add circles to the circle histogram
        circle_hist[yc_1[r_idx1], xc_1[r_idx1], r_idx1] += 1
        circle_hist[yc_2[r_idx2], xc_2[r_idx2], r_idx2] += 1

    # Find all the circles centers
    y, x, r = np.where(circle_hist > 11)
    circles = np.array([x, y, r + min_radius, circle_hist[y, x, r]]).T

    # Perform NMS
    circles = nms(circles, min_radius // 2)
    return circles


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float,
                               sigma_space: float) -> (np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """

    # OpenCV implement
    imgCV2 = cv2.bilateralFilter(in_image, k_size, sigma_color, sigma_space, borderType=cv2.BORDER_REPLICATE)

    # My implement
    imgMine = np.zeros(in_image.shape)

    half = k_size // 2
    w, h = in_image.shape

    # Padding
    img = cv2.copyMakeBorder(in_image, half + 1, half + 1, half + 1, half + 1, borderType=cv2.BORDER_REPLICATE)

    for i in range(w):
        for j in range(h):
            x, y = i + half, j + half
            pivot_v = img[x, y]
            window = img[x - half:x + half + 1, y - half:y + half + 1]

            # Weight by color
            color_gauss = np.exp(-np.power(pivot_v - window, 2) / (2 * sigma_color))

            # Weight by distance
            space_gauss = cv2.getGaussianKernel(int(k_size), sigma_space)
            space_gauss = space_gauss.dot(space_gauss.T)

            # Total Weight
            combo = color_gauss * space_gauss

            # Normalize
            pixel = combo * window / combo.sum()

            # Update
            imgMine[i, j] = int(pixel.sum())

    return imgCV2, imgMine
