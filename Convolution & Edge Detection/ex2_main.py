from ex2_utils_sol import *
import matplotlib.pyplot as plt
import time


def MSE(a: np.ndarray, b: np.ndarray) -> float:
    return np.square(a - b).mean()


def MAE(a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt(MSE(a, b)).mean()


def conv1Demo():
    signal = np.array([1.1, 1, 3, 4, 5, 6, 2, 1])
    kernel = np.array([1, 2, 2, 1])

    signal = np.array([1, 2, 2, 1])
    kernel = np.array([1, 1])

    sig_conv = conv1D(signal, kernel).astype(int)

    print("Signal:\t{}".format(signal))
    print("Numpy:\t{}".format(np.convolve(signal, kernel, 'full')))
    print("Mine:\t{}".format(sig_conv))


def conv2Demo():
    img = cv2.imread('input/beach.jpg', cv2.IMREAD_GRAYSCALE)

    kernel = np.ones((5, 5))
    kernel = kernel / kernel.sum()

    c_img = conv2D(img, kernel)
    cv_img = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)

    print("MSE: {:.2f}".format(MSE(c_img, cv_img)))
    print("Max Error: {:.2f}".format(np.abs((cv_img - c_img)).max()))

    f, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(c_img)
    ax[1].imshow(cv_img - c_img)
    ax[2].imshow(cv_img)
    plt.show()


def derivDemo():
    img = cv2.imread('input/beach.jpg', cv2.IMREAD_GRAYSCALE) / 255
    ori, mag = convDerivative(img)

    f, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].set_title('Ori')
    ax[1].set_title('Mag')
    ax[0].imshow(ori)
    ax[1].imshow(mag)
    plt.show()


def blurDemo():
    img = cv2.imread('input/beach.jpg', cv2.IMREAD_GRAYSCALE) / 255
    k_size = 5
    b1 = blurImage1(img, k_size)
    b2 = blurImage2(img, k_size)

    print("Blurring MSE: {:.6f}".format(np.sqrt(np.power(b1 - b2, 2).mean())))

    f, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(b1)
    ax[1].imshow(b1 - b2)
    ax[2].imshow(b2)
    plt.show()


def edgeDemoSimple():
    img = cv2.imread('input/codeMonkey.jpg', cv2.IMREAD_GRAYSCALE) / 255
    img = cv2.resize(img, (0, 0), fx=.25, fy=.25)
    edge_matrix = edgeDetectionZeroCrossingSimple(img)

    f, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].set_title("Ori")
    ax[1].set_title("Edge")
    ax[0].imshow(img)
    ax[1].imshow(edge_matrix)
    plt.show()


def edgeDemoLOG():
    img = cv2.imread('input/boxMan.jpg', cv2.IMREAD_GRAYSCALE) / 255
    img = cv2.resize(img, (0, 0), fx=.25, fy=.25)
    edge_matrix = edgeDetectionZeroCrossingLOG(img)

    f, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].set_title("Ori")
    ax[1].set_title("Edge")
    ax[0].imshow(img)
    ax[1].imshow(edge_matrix)
    plt.show()


def edgeDemo():
    edgeDemoSimple()
    edgeDemoLOG()


def houghDemo():
    img = cv2.imread('input/coins.jpg', cv2.IMREAD_GRAYSCALE) / 255
    min_r, max_r = 50, 100

    st = time.time()
    hough_rings = houghCircle(img, min_r, max_r)  # Mine
    print("Hough Time[Mine]: {:.3f} sec".format(time.time() - st))
    st = time.time()
    cv2_cir = cv2.HoughCircles((img * 255).astype(np.uint8), cv2.HOUGH_GRADIENT, 1, minDist=30, param1=500,
                               param2=80, minRadius=min_r, maxRadius=max_r)  # OpenCV
    print("Hough Time[CV]: {:.3f} sec".format(time.time() - st))

    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    for c in hough_rings:  # Mine
        circle1 = plt.Circle((c[0], c[1]), c[2], color='r', fill=False, linewidth=2)
        ax.add_artist(circle1)
    for c in cv2_cir[0]:  # OpenCV
        circle1 = plt.Circle((c[0], c[1]), c[2], color='b', fill=False, linewidth=1)
        ax.add_artist(circle1)
    plt.show()


def bilateralFilterDemo():
    img = cv2.imread('input/boxMan.jpg', cv2.IMREAD_GRAYSCALE)
    filtered_image_CV, filtered_image_my = bilateral_filter_implement(img, 9, 5.0, 1.5)

    f, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(filtered_image_CV)
    ax[0].set_title("OpenCV")
    ax[1].imshow(filtered_image_CV - filtered_image_my)
    ax[1].set_title("Diff")
    ax[2].imshow(filtered_image_my)
    ax[2].set_title("My")
    plt.show()

    print("MSE: {:.2f}".format(MSE(filtered_image_my, filtered_image_CV)))
    print("Max Error: {:.2f}".format(np.abs(filtered_image_my - filtered_image_CV).max()))


def main():
    print("ID:", myID())
    conv1Demo()
    conv2Demo()
    derivDemo()
    blurDemo()
    edgeDemo()
    houghDemo()
    bilateralFilterDemo()


if __name__ == '__main__':
    main()
