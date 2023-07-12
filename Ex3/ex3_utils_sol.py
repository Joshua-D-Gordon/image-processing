import math
from typing import List

import numpy as np
import cv2
import matplotlib.pyplot as plt


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 100


# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=20, win_size=5) -> (np.ndarray, np.ndarray):
    kernel = np.array([[-1, 0, 1]])
    window = int(win_size / 2)

    """ 
    Implement LK Algorithm
    for each point, calculate I_x, I_y, I_t 
    """
    fx_drive = cv2.filter2D(im2, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    fy_drive = cv2.filter2D(im2, -1, kernel.T, borderType=cv2.BORDER_REPLICATE)
    ft_drive = im2 - im1

    w, h = im1.shape
    pts = []
    uv = []

    for i in range(0, w, step_size):
        for j in range(0, h, step_size):
            Ix = fx_drive[i:i + window, j:j + window].flatten()
            Iy = fy_drive[i:i + window, j:j + window].flatten()
            It = ft_drive[i:i + window, j:j + window].flatten()
            AtA_ = [[(Ix * Ix).sum(), (Ix * Iy).sum()],
                    [(Ix * Iy).sum(), (Iy * Iy).sum()]]

            lambada = np.linalg.eigvals(AtA_)
            lambada1 = np.max(lambada)
            lambada2 = np.min(lambada)
            if lambada2 <= 1 or (lambada1 / lambada2) >= 100:
                pts.append([j, i])
                uv.append(np.array([0., 0.]))
            else:
                At = [[-(Ix * It).sum()], [-(Iy * It).sum()]]
                pts.append([j, i])
                ans = (np.linalg.inv(AtA_) @ At).reshape(2)
                uv.append(ans)
    return np.array(pts), np.array(uv)


def opticalFlowPyrLK(img1: np.ndarray, img2: np.ndarray, k: int, stepSize: int, winSize: int) -> np.ndarray:
    """
    :param img1: First image
    :param img2: Second image
    :param k: Pyramid depth
    :param stepSize: The image sample size
    :param winSize: The optical flow window size (odd number)
    :return: A 3d array, with a shape of (m, n, 2),
    where the first channel holds U, and the second V.
    """

    # Gaussian pyramid of size k for both images
    gaussianPyrImg1 = gaussianPyr(img1, k)
    gaussianPyrImg2 = gaussianPyr(img2, k)

    h, w = gaussianPyrImg1[-1].shape
    _, ans = opticalFlow(gaussianPyrImg1[- 1], gaussianPyrImg2[- 1], stepSize, winSize)
    ans = ans.reshape(h // stepSize, w // stepSize, 2)

    for i in range(1, k):
        ans *= 2
        expand = np.zeros((ans.shape[0] * 2, ans.shape[1] * 2, 2), dtype=ans.dtype)
        expand[::2, ::2] = ans
        ans = expand
        pts, uv = opticalFlow(gaussianPyrImg1[-i - 1], gaussianPyrImg2[-i - 1], stepSize, winSize)
        h, w = gaussianPyrImg1[-i - 1].shape
        ans += uv.reshape(h // stepSize, w // stepSize, 2)
    return ans


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """

    ret_list = [np.zeros(1)] * levels

    gaus_ker = cv2.getGaussianKernel(5, -1)
    gaus_ker = gaus_ker.dot(gaus_ker.T)
    gaus_ker = gaus_ker / gaus_ker.sum()

    pow_fac = np.power(2, levels)
    h, w = img.shape[:2]
    opt_h, opt_w = pow_fac * (h // pow_fac), pow_fac * (w // pow_fac)
    ret_list[0] = img[:opt_h, :opt_w]

    for lv in range(1, levels):
        b_img = cv2.filter2D(ret_list[lv - 1], -1, gaus_ker)
        b_img = b_img[::2, ::2]
        ret_list[lv] = b_img

    return ret_list


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """

    gas_ker = cv2.getGaussianKernel(5, -1)
    gas_ker = gas_ker.dot(gas_ker.T)
    gas_ker = gas_ker / gas_ker.sum()
    gas_ker_2 = gas_ker * 4
    gau_pyr = gaussianPyr(img, levels)

    ret_pyr = [np.zeros(1)] * levels
    ret_pyr[-1] = gau_pyr[-1]
    c = 1 if img.ndim < 3 else 3
    for lv in range(levels - 1):
        sml_img = gau_pyr[lv + 1]
        h, w = sml_img.shape[:2]
        exp_img = np.zeros((2 * h, 2 * w, c)).squeeze()
        exp_img[::2, ::2] = sml_img
        exp_img = cv2.filter2D(exp_img, -1, gas_ker_2)
        ret_pyr[lv] = gau_pyr[lv] - exp_img
    return ret_pyr


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Restores the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """

    gaus_ker = cv2.getGaussianKernel(5, -1)
    gaus_ker = gaus_ker.dot(gaus_ker.T)
    gaus_ker = gaus_ker / gaus_ker.sum()
    gaus_ker_2 = gaus_ker * 4

    rolling_img = lap_pyr[-1]
    for lv in range(len(lap_pyr) - 1, 0, -1):
        exp_img = imExpand(rolling_img, gaus_ker_2)
        rolling_img = exp_img + lap_pyr[lv - 1]
    return rolling_img


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray,
             mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """

    assert (img_1.shape == img_2.shape)
    assert (img_1.shape[:2] == mask.shape[:2])
    gas_ker = cv2.getGaussianKernel(5, -1)
    gas_ker = gas_ker.dot(gas_ker.T)
    gas_ker = gas_ker / gas_ker.sum()
    gas_ker_2 = gas_ker * 4

    h, w = img_1.shape[:2]

    pow_fac = np.power(2, levels)
    opt_h, opt_w = pow_fac * (h // pow_fac), pow_fac * (w // pow_fac)
    img_1 = img_1[:opt_h, :opt_w, :]
    img_2 = img_2[:opt_h, :opt_w, :]
    mask = mask[:opt_h, :opt_w, :]

    naive_blend = (img_1 * mask) + (1 - mask) * img_2
    lap1 = laplaceianReduce(img_1, levels)
    lap2 = laplaceianReduce(img_2, levels)
    mask_pyr = gaussianPyr(mask, levels)

    pyr_blend = lap1[-1] * mask_pyr[-1] + (1 - mask_pyr[-1]) * lap2[-1]
    for lv in range(levels - 1, 0, -1):
        exp_img_p = imExpand(pyr_blend, gas_ker_2)
        mask = mask_pyr[lv - 1]
        exp_img1 = lap1[lv - 1]
        exp_img2 = lap2[lv - 1]

        exp_img = exp_img1 * mask + exp_img2 * (1 - mask)
        pyr_blend = exp_img + exp_img_p

    return naive_blend, pyr_blend


# ---------------------------------------------------------------------------
# --------------------------- SELF IMPLEMENTATION ---------------------------
# ---------------------------------------------------------------------------


def calculateLK(im1: np.ndarray, im2: np.ndarray, points: np.ndarray, win_size=5) -> (np.ndarray, np.ndarray):
    """
    Calculates the optical flow using Lukas-Kande algorithm
    :param im1: First Image
    :param im2: Second Image
    :param points: Points to calculate the optical flow
    :param win_size: 
    :return: List of [x,y,[uv],mag], the location of the pixel in the first image, 
    and the movement (uv) to its matching pixel, and the image magnification
    """

    Ix = cv2.Sobel(im2, cv2.CV_64F, 1, 0, ksize=3, scale=1 / 8)
    Iy = cv2.Sobel(im2, cv2.CV_64F, 0, 1, ksize=3, scale=1 / 8)
    It = (im2 - im1)

    window_size = win_size
    w_2 = window_size // 2
    h, w = im1.shape[:2]

    uv_lst = []
    pts = []
    points = points[points[:, 0] > w_2, :]
    points = points[points[:, 0] < h - w_2, :]
    points = points[points[:, 1] > w_2, :]
    points = points[points[:, 1] < w - w_2, :]
    for id_y, id_x in points:
        c_ix = Ix[id_y - w_2:id_y + w_2 + 1, id_x - w_2:id_x + w_2 + 1].ravel()
        c_iy = Iy[id_y - w_2:id_y + w_2 + 1, id_x - w_2:id_x + w_2 + 1].ravel()
        c_it = It[id_y - w_2:id_y + w_2 + 1, id_x - w_2:id_x + w_2 + 1].ravel()

        A = np.array([c_ix, c_iy])
        # Invertible check
        ATA = A @ A.T
        eig1, eig2 = np.linalg.eigvals(ATA)
        # Small eig. values check
        if eig1 < 10 or eig1 / eig2 > 2:
            continue

        b = -c_it.T
        ATb = A @ (b)
        uv = np.linalg.inv(ATA) @ (ATb)

        uv_lst.append(uv)
        pts.append((id_x, id_y))

    return np.array(pts), np.array(uv_lst)


def displayOpticalFlow(img: np.ndarray, pts: np.ndarray, uvs: np.ndarray):
    plt.imshow(img, cmap='gray')
    plt.quiver(pts[:, 0], pts[:, 1], uvs[:, 0], uvs[:, 1], color='r')

    plt.show()


def imExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    """
    Expands a Gaussian pyramid level one step up
    :param img: Pyramid image at a certain level
    :param gs_k: The kernel to use in expanding
    :return: The expanded level
    """

    h, w = img.shape[:2]
    c = 1 if img.ndim < 3 else 3
    exp_img = np.zeros((h * 2, w * 2, c)).squeeze()
    exp_img[::2, ::2] = img
    exp_img = cv2.filter2D(exp_img, -1, gs_k)
    return exp_img
