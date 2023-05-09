import numpy as np
from utils import *


@timing_decorator
def serial_apply_tf(tf: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Args:
        tf: (4, 4) - homogeneous transformation
        points: (N, 3) - x, y, z
    
    Returns:
        transformed_points: (N, 3) - x, y, z
    """
    num_points = points.shape[0]
    transformed_points = np.zeros((num_points, 3))
    for p_idx in range(num_points):
        x, y, z = points[p_idx]
        transformed_xyz1 = tf @ np.array([x, y, z, 1]).reshape(-1, 1)  # (4, 1)
        transformed_points[p_idx] = transformed_xyz1[:3, 0]
    
    return transformed_points


@timing_decorator
def apply_tf_v1(tf: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Args:
        tf: (4, 4) - homogeneous transformation
        points: (N, 3) - x, y, z
    
    Returns:
        transformed_points: (N, 3) - x, y, z
    """
    # convert points to homogeneous coordinate
    points_ = np.pad(points, pad_width=[(0, 0), (0, 1)], constant_values=1.0)  # (N, 4) - x, y, z, 1

    tformed_points = points_ @ tf.T
    return tformed_points[:, :3]


@timing_decorator
def apply_tf_v2(tf: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Args:
        tf: (4, 4) - homogeneous transformation
        points: (N, 3) - x, y, z
    
    Returns:
        transformed_points: (N, 3) - x, y, z
    """
    # skip converting points to homogeneous coord

    tformed_points = points @ tf[:3, :3].T + tf[:3, -1]
    return tformed_points


def main():
    points = np.fromfile('/home/user/Downloads/kitti/kitti_raw/2011_09_26/2011_09_26_drive_0036_sync/velodyne_points/data/0000000101.bin',
                         dtype=np.float32).reshape(-1, 4)[:, :3]
    print('points: ', points.shape)
    show_point_cloud(points)

    # remove points outside range
    pc_range = np.array([-51.2, -51.2, -3., 51.2, 51.2, 1.0])
    points = points[np.logical_and(np.all(points > pc_range[:3], axis=1), np.all(points < pc_range[3:], axis=1))]
    show_point_cloud(points)

    angle = np.deg2rad(45)
    cos, sin = np.cos(angle), np.sin(angle)
    tx, ty, tz = [10., 5., 0.]
    tf = np.array([
        [cos,   -sin,   0,      tx],
        [sin,   cos,    0,      ty],
        [0,     0,      1,      tz],
        [0,     0,      0,      1]
    ])

    tformed_points = serial_apply_tf(tf, points)
    show_point_cloud(tformed_points)

    tformed_points_v1 = apply_tf_v1(tf, points)
    assert np.isclose(tformed_points, tformed_points_v1).all()

    tformed_points_v2 = apply_tf_v2(tf, points)
    assert np.isclose(tformed_points, tformed_points_v2).all()


if __name__ == '__main__':
    main()
