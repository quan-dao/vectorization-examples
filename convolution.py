import numpy as np


img = np.arange(3 * 4).reshape(3, 4)

img = np.array([
    [0, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0],
])
num_rows, num_cols = img.shape

print('img:\n', img)

kernel = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
],dtype=float)

kernel = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1]
],dtype=float)

conv_kernel_size = 3
num_valid_rows = num_rows - int(conv_kernel_size // 2) * 2
num_valid_cols = num_cols - int(conv_kernel_size // 2) * 2

rows = np.arange(num_valid_rows) + int(conv_kernel_size // 2)
cols = np.arange(num_valid_cols) + int(conv_kernel_size // 2)

cc, rr = np.meshgrid(cols, rows)
print('rr:\n', rr)
print('cc:\n', cc)

rr = rr.reshape(-1)  # (N,)
cc = cc.reshape(-1)  # (N,)
num_pads = rr.shape[0]

neighborhood_r = np.array([-1, -1, -1, 
                           0, 0, 0, 
                           1, 1, 1])
neighborhood_c = np.array([-1, 0, 1,
                           -1, 0, 1,
                           -1, 0, 1])

pad_indices_r = rr.reshape(-1, 1) + neighborhood_r  # (N, 9)
pad_indices_c = cc.reshape(-1, 1) + neighborhood_c  # (N, 9)

pad = img[pad_indices_r.reshape(-1), pad_indices_c.reshape(-1)].reshape(num_pads, -1)  # (N * 9) -> (N, 9)
print('pad: ', pad)

conv_img = np.sum(pad * kernel.reshape(-1), axis=-1)  # (N, 9) -> (N,)
conv_img = conv_img.reshape(num_valid_rows, num_valid_cols)
print('conv_img:\n', conv_img)
