import numpy as np
from typing import Tuple
import cv2
import matplotlib.pyplot as plt

from utils import timing_decorator


def find_pad_centers(img_hw: tuple, kernel_size: int) -> Tuple[np.ndarray]:
    """
    Args:
        img_hw:
        kernel_size:
    
    Returns:
        centers_row_index: (N,)
        centers_col_index: (N,)
    """
    num_rows, num_cols = img_hw
    offset = int(kernel_size // 2)
    num_valid_rows = num_rows - offset * 2
    num_valid_cols = num_cols - offset * 2

    rows = np.arange(num_valid_rows) + offset
    cols = np.arange(num_valid_cols) + offset

    return rows, cols


@timing_decorator
def serial_conv(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Args:
        img: (H, W)
        kernel: (S, S) - square kernel
    
    Returns:
        conv_img: (valid_H, valid_W)
    """
    def _conv(pad: np.ndarray, kernel: np.ndarray):
        assert pad.shape == kernel.shape
        assert len(pad.shape) == 1
        out = 0
        for i in range(pad.shape[0]):
            out += (pad[i] * kernel[i])
        return out


    centers_row, centers_col = find_pad_centers(img.shape, kernel.shape[0])  # (N_row,), (N_col,)
    offset = int(kernel.shape[0] // 2)
    conv_img = np.zeros((centers_row.shape[0], centers_col.shape[0]))
    
    # define neighbors' coord in pad's local coord
    neighbors_row_unique = np.arange(start=-offset, stop=offset + 1)
    neighbors_col_unique = np.arange(start=-offset, stop=offset + 1)
    neighbors_row, neighbors_col = list(), list()
    for ngh_row in neighbors_row_unique:
        for ngh_col in neighbors_col_unique:
            neighbors_row.append(ngh_row)
            neighbors_col.append(ngh_col)
    neighbors_row = np.array(neighbors_row)  # (S**2,)
    neighbors_col = np.array(neighbors_col)  # (S**2,)

    kernel_flat = kernel.reshape(-1)

    for crow in centers_row:
        for ccol in centers_col:
            # map neighbors' coord from local to global (i.e., img) coord
            rows = neighbors_row + crow
            cols = neighbors_col + ccol

            # extract img pad
            pad = img[rows, cols]  # (S**2,)

            # perform conv
            conv_img[crow - offset, ccol - offset] = _conv(pad, kernel_flat)
    
    return conv_img


@timing_decorator
def conv(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Args:
        img: (H, W)
        kernel: (S, S) - square kernel
    
    Returns:
        conv_img: (valid_H, valid_W)
    """
    centers_row, centers_col = find_pad_centers(img.shape, kernel.shape[0])  # (N_row,), (N_col,)
    offset = int(kernel.shape[0] // 2)
    
    # pads' center in img coord
    cc, rr = np.meshgrid(centers_col, centers_row)
    rr = rr.reshape(-1)  # (N,)
    cc = cc.reshape(-1)  # (N,)
    num_pads = rr.shape[0]

    # local coord
    neighbors_row_unique = np.arange(start=-offset, stop=offset + 1)
    neighbors_col_unique = np.arange(start=-offset, stop=offset + 1)

    neighborhood_c, neighborhood_r = np.meshgrid(neighbors_col_unique, neighbors_row_unique)

    # img coord
    pad_indices_r = rr.reshape(-1, 1) + neighborhood_r.reshape(-1)  # (N, S**2)
    pad_indices_c = cc.reshape(-1, 1) + neighborhood_c.reshape(-1)  # (N, S**2)

    # extract pad
    pad = img[pad_indices_r.reshape(-1), pad_indices_c.reshape(-1)].reshape(num_pads, -1)  # (N * 9) -> (N, 9)

    # conv
    conv_img = np.sum(pad * kernel.reshape(-1), axis=-1)  # (N, 9) -> (N,)
    conv_img = conv_img.reshape(centers_row.shape[0], centers_col.shape[0])

    return conv_img


if __name__ == '__main__':
    img = np.arange(3 * 4).reshape(3, 4)
    kernel = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ],dtype=float)

    img = np.array([
        [0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
    ])
    kernel = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1]
    ],dtype=float)


    img_color = cv2.imread('music.png', -1)
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ],dtype=float)
    
    # conv_img = serial_conv(img, kernel)
    conv_img = conv(img, kernel)

    print('conv_img:\n', conv_img)

    # assert np.isclose(conv_img, _conv_img).all()

    fig, axe = plt.subplots(1, 2)
    axe[0].imshow(img_color)
    axe[0].set_title('input')
    axe[1].imshow(conv_img, cmap='gray')
    axe[1].set_title('vertical edges')
    plt.show()
