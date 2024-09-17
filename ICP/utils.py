import numpy as np


def depth_to_point_cloud(K, depth_image):
    """
    back project a depth image to a point cloud
    args:
        K (numpy.array [3, 3])
        depth_image (numpy.array [h, w]): each entry is a z depth value
    Returns:
        numpy.array [n, 3]: each row represents a different valid 3D point
    """
    fx, fy = K[0,0], K[1,1] 
    cx, cy = K[0,2], K[1,2]
    h, w = depth_image.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u, v = u.flatten(u), u.flatten(v)
    Z = depth_image.flatten()
    valid = Z > 0
    u, v, Z = u[valid], v[valid], Z[valid]
    X, Y = (u - cx) * Z/fx, (v - cy) * Z/fy
    points = np.vstack((X, Y, Z)).T
    return points    