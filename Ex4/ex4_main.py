import os
import cv2
from ex4_utils_sol import *



def displayDepthImage(l_img, r_img, disparity_range=(0, 5), method=disparitySSD):
    p_size = 5
    d_ssd = method(l_img, r_img, disparity_range, p_size)
    plt.matshow(d_ssd)
    plt.colorbar()
    plt.show()


def main():
    # Print your ID number
    print("ID:", 100)

    # Read images
    i = 0
    L = cv2.imread(os.path.join('input', 'pair%d-L.png' % i), 0) / 255.0
    R = cv2.imread(os.path.join('input', 'pair%d-R.png' % i), 0) / 255.0

    # Display depth SSD
    displayDepthImage(L, R, (0, 4), method=disparitySSD)

    # Display depth NC
    displayDepthImage(L, R, (0, 4), method=disparityNC)

    """
    input:
        src pnt = np.array([[279, 552],[372, 559],[362, 472],[277, 469]])
        dst pnt = np.array([[24, 566],[114, 552],[106, 474],[19, 481]])
    output:
        M = [[ 1.84928168e+00 6.29286954e-01 2.00796141e+02]
            [ 9.37137905e-02 3.12582854e+00 -5.89220129e+02]
            [-5.32787017e-04 2.02578161e-03 1.00000000e+00]]
        Error: 1.815947868067344
    """

    src = np.array([[279, 552],
                    [372, 559],
                    [362, 472],
                    [277, 469]])
    dst = np.array([[24, 566],
                    [114, 552],
                    [106, 474],
                    [19, 481]])
    h, error = computeHomography(src, dst)

    print(h, "\n", error)

    dst = cv2.imread(os.path.join('input', 'billBoard.jpg'))[:, :, [2, 1, 0]] / 255.0
    src = cv2.imread(os.path.join('input', 'car.jpg'))[:, :, [2, 1, 0]] / 255.0

    warpImag(src, dst)


if __name__ == '__main__':
    main()
