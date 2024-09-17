# tsdf.py
import cv2
import time
import numpy as np
from skimage import measure
from utils import camera_to_image, world_to_camera, Ply

class TSDFVolume:
    def __init__(self, volume_bounds, voxel_size):
        """
        args:
            volume_bounds (numpy.array [3, 2]): rows index [x, y, z] and cols index [min_bound, max_bound]. Units are in meters.
            voxel_size (float): The side length of each voxel in meters.
        """
        # raise value error if not conformant
        if volume_bounds.shape != (3,2):
            raise ValueError("Incorrect Shape")
        if voxel_size <= 0.0:
            raise ValueError("Invalid Size")
            
        """Numpy array [3, 2] of float32s, where rows index [x, y, z] and cols index [min_bound, max_bound]. Units are in meters."""
        self._volume_bounds = volume_bounds
        
        """float side length in meters of each 3D voxel cube."""
        self._voxel_size = float(voxel_size)
        
        """float tsdf truncation margin, the max allowable distance away from a surface in meters."""
        self._truncation_margin = 2 * self._voxel_size
        # adjust volume bounds
        self.voxel_bounds = np.ceil((self._volume_bounds[:,1]-self._volume_bounds[:,0])/self._voxel_size).copy(order='C').astype(int) 
        self._volume_bounds[:, 1] = self._volume_bounds[:, 0] + (self.voxel_bounds * self._voxel_size)
        
        """Origin of the voxel grid in world coordinate. Units are in meters."""
        self._volume_origin = self._volume_bounds[:,0].copy(order='C').astype(np.float32)
        print(f'Voxel volume size: {self.voxel_bounds[0]} x {self.voxel_bounds[1]} x {self.voxel_bounds[2]} - # voxels: {self.voxel_bounds[0] * self.voxel_bounds[1] * self.voxel_bounds[2]}')
        
        """Numpy array of float32s representing tsdf volume where each voxel represents a volume self._voxel_size^3. Shape of this volume is determined by (max_bound - min_bound)/ self._voxel_size. Each entry contains the distance to the nearest surface in meters, truncated by self._truncation_margin."""
        self._tsdf_volume = np.ones(self.voxel_bounds).astype(np.float32)
        
        """Numpy array of float32s with shape [self._tsdf_volume.shape, 3] in range [0.0, 255.0]. So each entry in the volume contains the average r, g, b color."""
        self._color_volume = np.zeros(np.append(self.voxel_bounds, 3)).astype(np.float32)
        
        """Numpy array [number of voxels, 3] of uint8s. [[0, 0, 0], [0, 0, 1], ...,]. When a new observation is made, we need to determine which of these voxel coordinates is "valid" so we can decide what voxels to update."""
        xv, yv, zv = np.meshgrid(
            range(self.voxel_bounds[0]),
            range(self.voxel_bounds[1]),
            range(self.voxel_bounds[2]),
        )
        self._voxel_coords = np.vstack((xv.flatten(), yv.flatten(), zv.flatten())).astype(int).T


    def get_mesh(self):
        """
        Reconstructs a 3D mesh from a tsdf volume using the marching cubes algorithm
        Returns:
            numpy.array [n, 3]: each row represents a 3D point.
            numpy.array [k, 3]: each row is a list of point indices used to render triangles.
            numpy.array [n, 3]: each row represents the normal vector for the corresponding 3D point.
            numpy.array [n, 3]: each row represents the color of the corresponding 3D point.
        """
        # Marching cubes
        voxel_points, triangles, normals, _ = measure.marching_cubes(self._tsdf_volume, level=0, method='lewiner')
        points_ind = np.round(voxel_points).astype(int)
        points = self.voxel_to_world(self._volume_origin, voxel_points, self._voxel_size)
        
        # Get vertex colors
        rgb_vals = self._color_volume[points_ind[:, 0], points_ind[:, 1], points_ind[:, 2]]
        colors_r, colors_g, colors_b = rgb_vals[:, 0], rgb_vals[:, 1], rgb_vals[:, 2]
        colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
        colors = colors.astype(np.uint8)
        return points, triangles, normals, colors


    def get_valid_points(self, depth_image, voxel_u, voxel_v, voxel_z):
        """Compute a boolean array for indexing the voxel volume. Note that every time the method integrate(...) is called, not every voxel in the volume will be updated. This method returns a boolean matrix called valid_points with dimension (n, ), where n = # of voxels. Index i of valid_points will be true if this voxel will be updated, false if the voxel needs not to be updated."""
        image_height, image_width = depth_image.shape
        # Ensure voxel_u and voxel_v are valid integer indices
        voxel_u = voxel_u.astype(int)
        voxel_v = voxel_v.astype(int)
    
        # Eliminate pixels not in the image bounds or that are behind the image plane
        valid_pixels = np.logical_and(voxel_u >= 0,
                                      np.logical_and(voxel_u < image_width,
                                      np.logical_and(voxel_v >= 0,
                                      np.logical_and(voxel_v < image_height, voxel_z > 0))))
    
        # Get depths for valid coordinates u, v from the depth image. Zero elsewhere.
        depths = np.zeros(voxel_u.shape)
        depths[valid_pixels] = depth_image[voxel_v[valid_pixels], voxel_u[valid_pixels]]
    
        # Filter out zero depth values and depth + truncation margin >= voxel_z
        valid_points = np.logical_and(depths > 0, depths+self._truncation_margin >= voxel_z)
        return valid_points
        
    
    @staticmethod
    def voxel_to_world(volume_origin, voxel_size, voxel_coords):
        """Convert from voxel coordinates to world coordinates"""
        return volume_origin + voxel_coords * voxel_size
        

    @staticmethod
    def compute_tsdf(depth_image, voxel_z, truncation_margin, valid_points, valid_pixels):
        """Compute the new TSDF value for each valid point. We apply truncation and normalization in the end, so that tsdf value is in the range [-1,1]."""
        Z = voxel_z[valid_points]
        proj = np.zeros_like(Z)
        
        for idx, pixel in enumerate(valid_pixels):
            u, v = pixel[0], pixel[1]
            depth = depth_image[v, u]
            proj[idx] = depth - Z[idx]
            
        for idx, dist in enumerate(proj):
            proj[idx] = max(-1, min(1, dist/truncation_margin))

        tsdf = proj
        return tsdf


    @staticmethod
    def update_tsdf(tsdf_old, tsdf_new, color_old, color_new):
        """
        Update the TSDF value and color for the voxels that have new observations. 
        We only update the tsdf and color value when the new absolute value of tsdf_new[i] is smaller than that of tsdf_old[i].
        """
        for idx in range(len(tsdf_new)):
            if abs(tsdf_new[idx]) < abs(tsdf_old[idx]):
                tsdf_old[idx] = tsdf_new[idx]
                color_old[idx, :] = color_new[idx, :]
        return tsdf_old, color_old


    def integrate(self, color_image, depth_image, camera_intrinsics, camera_pose):
        """
        Integrate an RGB-D observation into the TSDF volume, by updating the tsdf volume, and color volume.
        Args:
            color_image (numpy.array [h, w, 3]): rgb image.
            depth_image (numpy.array [h, w]): 'z' depth image.
            camera_intrinsics (numpy.array [3, 3]): given as [[fu, 0, u0], [0, fv, v0], [0, 0, 1]]
            camera_pose (numpy.array [4, 4]): SE3 transform representing pose (camera to world)
        """
        color_image = color_image.astype(np.float32)
        depth_image = depth_image.astype(np.float32)
        voxel_world_coords = self.voxel_to_world(self._volume_origin, self._voxel_size, self._voxel_coords)
        voxel_camera_coords = world_to_camera(camera_pose, voxel_world_coords)
        voxel_img_coords = camera_to_image(camera_intrinsics, voxel_camera_coords)
        voxel_u, voxel_v, voxel_z = voxel_img_coords[:,0], voxel_img_coords[:,1], voxel_camera_coords[:,2]
        valid_points = self.get_valid_points(depth_image, voxel_u, voxel_v, voxel_z)
        valid_voxels = self._voxel_coords[valid_points]
        valid_pixels = voxel_img_coords[valid_points]
        tsdf = self.compute_tsdf(depth_image, voxel_z, self._truncation_margin, valid_points, valid_pixels)

        tsdf_old  = self._tsdf_volume[valid_voxels[:,0], valid_voxels[:,1], valid_voxels[:,2]]
        color_old = self._color_volume[valid_voxels[:,0], valid_voxels[:,1], valid_voxels[:,2]]
        color_new = color_image[valid_pixels[:,1].astype(int), valid_pixels[:,0].astype(int)]
        
        tsdf_updated, color_updated = self.update_tsdf(tsdf_old, tsdf, color_old, color_new)
        self._tsdf_volume[valid_voxels[:,0], valid_voxels[:,1], valid_voxels[:,2]] = tsdf_updated
        self._color_volume[valid_voxels[:,0], valid_voxels[:,1], valid_voxels[:,2]] = color_updated


if __name__ == "__main__":
    image_count = 10
    voxel_size = 0.01
    volume_bounds = np.array([[-0.75, 0.75], [-0.75,0.75], [0.,0.8]])
    camera_intrinsics = np.loadtxt("./data/camera-intrinsics.txt")
    tsdf_volume = TSDFVolume(volume_bounds, voxel_size)
    
    start_time = time.time()
    for i in range(image_count):
        print(f"Fusing frame {i+1}/{image_count}")
    
        # Load RGB-D image and camera pose
        color_image = cv2.imread(f"./data/frame-{i:06d}.color.png")
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        depth_image = cv2.imread(f"./data/frame-{i:06d}.depth.png", -1).astype(float) / 1000.0
        camera_pose = np.loadtxt(f"./data/frame-{i:06d}.pose.txt")
    
        # Integrate observation into voxel volume
        tsdf_volume.integrate(color_image, depth_image, camera_intrinsics, camera_pose)

    fps = image_count / (time.time() - start_time)
    print("Average FPS: {:.2f}".format(fps))

    # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving mesh to mesh.ply...")
    points, faces, normals, colors = tsdf_volume.get_mesh()
    mesh = Ply(triangles=faces, points=points, normals=normals, colors=colors)
    mesh.write('mesh_output.ply')
    print("Done")

    