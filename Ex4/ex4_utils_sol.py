import numpy as np
import matplotlib.pyplot as plt


def matchSSD(y_pos, x_pos, patch, R, disparity_range):
    p_size = len(patch)
    disp_diff = disparity_range[1] - disparity_range[0]
    if x_pos + disp_diff + p_size >= R.shape[1]:
        disp_diff = -x_pos - p_size + R.shape[1]
    start_x = x_pos + disparity_range[0]
    start_y = y_pos - p_size // 2
    end_y = y_pos + p_size // 2 + 1

    ssd_strip = np.zeros(disp_diff)
    for idx in range(disp_diff):
        c_patch = R[start_y:end_y,
                  start_x + idx - p_size // 2: start_x + idx + p_size // 2 + 1]
        ssd_strip[idx] = np.square(patch - c_patch).sum()

    return np.argmin(ssd_strip) + disparity_range[0]


def disparitySSD(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: Minimum and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """

    h, w = img_l.shape
    disp_map = np.zeros(img_l.shape)
    for y_pos in range(k_size // 2 + 1, h - k_size):
        for x_pos in range(k_size // 2 + 1, w - max(k_size, disp_range[1] + 1)):
            start_x = x_pos - k_size // 2
            end_x = x_pos + k_size // 2 + 1
            start_y = y_pos - k_size // 2
            end_y = y_pos + k_size // 2 + 1
            patch = img_r[start_y:end_y, start_x:end_x]

            disp_map[y_pos, x_pos] = matchSSD(y_pos, x_pos, patch, img_l, disp_range)

    return disp_map


def matchNC(y_pos, x_pos, patch, R, disparity_range):
    p_size = len(patch)
    disp_diff = disparity_range[1] - disparity_range[0]
    if x_pos + disp_diff + p_size >= R.shape[1]:
        disp_diff = -x_pos - p_size + R.shape[1]
    start_x = x_pos + disparity_range[0]
    start_y = y_pos - p_size // 2
    end_y = y_pos + p_size // 2 + 1

    p_std = np.std(patch)
    nc_strip = np.zeros(disp_diff)
    for idx in range(disp_diff):
        c_patch = R[start_y:end_y, start_x + idx - p_size // 2:start_x + idx + p_size // 2 + 1]
        c_std = np.std(c_patch)
        nc_strip[idx] = (patch * c_patch / (c_std * p_std + np.finfo('float').eps)).mean()

    # return np.argmax(conv_strip[y_offset, :])
    return np.argmax(nc_strip) + disparity_range[0]


def disparityNC(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: Minimum and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """

    h, w = img_l.shape
    disp_map = np.zeros(img_l.shape)
    for y_pos in range(k_size // 2 + 2, h - k_size):
        for x_pos in range(k_size // 2 + 2, w - max(k_size, disp_range[1] + 1)):
            start_x = x_pos - k_size // 2
            end_x = x_pos + k_size // 2 + 1
            start_y = y_pos - k_size // 2
            end_y = y_pos + k_size // 2 + 1
            patch = img_r[start_y:end_y, start_x:end_x]

            disp_map[y_pos, x_pos] = matchNC(y_pos, x_pos, patch, img_l, disp_range)

    return disp_map


def computeHomography(src_pnt: np.ndarray, dst_pnt: np.ndarray) -> (np.ndarray, float):
    """
        Finds the homography matrix, M, that transforms points from src_pnt to dst_pnt.
        returns the homography and the error between the transformed points to their
        destination (matched) points.

        Error = np.sqrt(sum((M.dot(src_pnt)-dst_pnt)**2))

        src_pnt: 4+ keypoints locations (x,y) on the original image. Shape:[4+,2]
        dst_pnt: 4+ keypoints locations (x,y) on the destination image. Shape:[4+,2]

        return: (Homography matrix shape:[3,3], Homography error)
    """

    n = src_pnt.shape[0]
    a_mat = np.zeros((2 * n, 9))

    for r in range(n):
        x, y = src_pnt[r]
        x_t, y_t = dst_pnt[r]
        a_mat[2 * r, :] = [x, y, 1, 0, 0, 0, -x_t * x, -x_t * y, -x_t]
        a_mat[2 * r + 1, :] = [0, 0, 0, x, y, 1, -y_t * x, -y_t * y, -y_t]

    U, D, Vh = np.linalg.svd(a_mat)
    H = Vh[-1, :].reshape(3, 3) / Vh[-1, -1]

    # Error estimation
    h_src = Homogeneous(src_pnt)
    pred = H.dot(h_src.T).T
    pred = unHomogeneous(pred)
    error = np.sqrt(np.square(pred - dst_pnt).mean())

    return H, error


def Homogeneous(a):
    return np.concatenate((a, np.ones((len(a), 1))), axis=1)


def unHomogeneous(a):
    return a[:, :2] / a[:, 2:]


def warpImag(src_img: np.ndarray, dst_img: np.ndarray) -> None:
    """
       Displays both images, and lets the user mark 4 or more points on each image.
       Then calculates the homography and transforms the source image on to the destination image.
       Then transforms the source image onto the destination image and displays the result.

       src_img: The image that will be 'pasted' onto the destination image.
       dst_img: The image that the source image will be 'pasted' on.

       output: None.
    """

    dst_p = []
    fig1 = plt.figure()

    def onclick_1(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, '*r')
        dst_p.append([x, y])

        if len(dst_p) == 4:
            plt.close()
        plt.show()

    # display image 1
    cid = fig1.canvas.mpl_connect('button_press_event', onclick_1)
    plt.imshow(dst_img)
    plt.show()
    dst_p = np.array(dst_p)

    ##### Your Code Here ######
    dst_h, dst_w = dst_img.shape[:2]
    src_h, src_w = src_img.shape[:2]
    src_p = np.array([
        [0, 0],
        [src_w, 0],
        [src_w, src_h],
        [0, src_h],
    ])

    H, _ = computeHomography(dst_p, src_p)

    X, Y = np.meshgrid(range(dst_w), range(dst_h))
    XY1 = np.ones_like(dst_img)
    XY1[:, :, 0] = X
    XY1[:, :, 1] = Y
    XY1 = XY1.reshape((dst_h * dst_w, 3))

    XY2 = XY1.dot(H.T)
    XY2 = (XY2[:, :2] / XY2[:, 2:]).astype(int)
    XY1 = XY1.astype(int)

    XY1 = XY1[XY2[:, :2].min(axis=1) >= 0]
    XY2 = XY2[XY2[:, :2].min(axis=1) >= 0]

    XY1 = XY1[XY2[:, 0] < src_w]
    XY2 = XY2[XY2[:, 0] < src_w]

    XY1 = XY1[XY2[:, 1] < src_h]
    XY2 = XY2[XY2[:, 1] < src_h]

    src_out = np.zeros_like(dst_img)
    src_out[XY1[:, 1], XY1[:, 0]] = src_img[XY2[:, 1], XY2[:, 0]]

    mask = np.ones_like(dst_img)
    mask[XY1[:, 1], XY1[:, 0]] = 0

    out = dst_img * mask + src_out * (1 - mask)
    plt.imshow(out)
    plt.show()
