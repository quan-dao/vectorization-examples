import numpy as np
from utils import timing_decorator


@timing_decorator
def serial_distance(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """
    Build a distance matrix D, of size N1 x N2, such that D[i, j] = ||pts1[i] - pts2[j]||
    Args:
        pts1: (N1, 3) - x, y, z
        pts2: (N2, 3) - x, y, z
    
    Returns:
        dist_mat: (N1, N2)
    """
    num_pts1, num_pts2 = pts1.shape[0], pts2.shape[0]
    dist_mat = np.zeros((num_pts1, num_pts2))
    for p1_idx in range(num_pts1):
        for p2_idx in range(num_pts2):
            dist_mat[p1_idx, p2_idx] = np.linalg.norm(pts1[p1_idx] - pts2[p2_idx])
    
    return dist_mat


@timing_decorator
def distance(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """
    Build a distance matrix D, of size N1 x N2, such that D[i, j] = ||pts1[i] - pts2[j]||
    Args:
        pts1: (N1, 3) - x, y, z
        pts2: (N2, 3) - x, y, z
    
    Returns:
        dist_mat: (N1, N2)
    """
    dist = pts1[:, np.newaxis] - pts2[np.newaxis, :]  # (N1, N2, 3)
    dist_mat = np.linalg.norm(dist, axis=-1)
    return dist_mat 


if __name__ == '__main__':
    num_pts1 = 100
    num_pts2 = 200
    pts1 = np.random.rand(num_pts1, 3)
    pts2 = np.random.rand(num_pts2, 3)

    seraial_d = serial_distance(pts1, pts2)
    d = distance(pts1, pts2)

    assert np.isclose(seraial_d, d).all()
