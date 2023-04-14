import numpy as np
import open3d as o3d
import time


def show_point_cloud(points: np.ndarray) -> None:
    """
    Args:
        points: (N, 3) - x, y, z
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    lidar_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0, origin=np.zeros(3))
    o3d.visualization.draw_geometries([pcd, lidar_frame])


def timing_decorator(func):
    def timer_wrapper(*args, **kwargs):
        tic = time.time()
        func_out = func(*args, **kwargs)
        tac = time.time()
        print(f"{func.__name__!r} takes {tac - tic} sec")
        return func_out
    return timer_wrapper



