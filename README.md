# Hidden spaces module
This module visualizes which areas of a buildings interior are not scanned in a volumetric fashion using voxels
to represent 3D space.
___
## How it works
The identification of the volumes (voxels) that are not visited by the explorers is done gradually by processing one 
Keyframe at a time and updating the status of the voxels. The voxels can have 6 different states (statuses):
1. **Mesh** -> Voxels occupied by the buildings hull
2. **Exterior** -> Voxels outside of the building
3. **Scanned** -> Voxels that contain points from the point cloud
4. **Seen** -> Voxels that represent the void space between the camera's position and the Scanned voxels
5. **Unknown** -> Initially they represent the interior of the building. At the end of the process they represent the voxels that remain unknown (i.e., are not Scanned or Seen)
6. **Temporary** -> Used for some extreme cases where the camera lies outside of the voxel grid, for isntance in the beginning of the scan.

This process is explained 
in [steps](#steps).

### Inputs
1. Building's hull as triangular mesh
2. Point cloud

#### Preprocess of the inputs
The building hull is manually transformed so that it aligns with X, Y, and Z axes using CloudCompare and 
is stored as a new file.

The point cloud is manually transformed so that it fits in the building's hull using CloudCompare and the transformation
matrix is stored for further use.

### Steps
1. The building's hull is voxelized using `open3d.geometry.VoxelGrid.create_from_triangle_mesh()`.
2. The voxels are transferred in a custom [`VoxelGrid`](#voxelgrid-class), and are marked as **occupied**.
3. The voxel grid is the densified, which means that all the voxels in the domain are created and are marked as unknown.
4. The **exterior** voxels (i.e., the voxels outside of the building) are marked using 6 neighbours region growing.
5. Mark **scanned** and **void** voxels
   1. For each Keyframe (pair of RGB + Depth images) that is retrieved from the database a point cloud is generated and transformed using the transformation matrix retrieved in the [preprocessing](#preprocess-of-the-inputs).
   2. This point cloud is then voxelized using `open3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds()`.
   3. This voxel grid has the same origin as the one in step 2, and its voxels are copied to the latter and marked as **scanned**.
   4. The voxels between the camera's position and the voxels generated by the point-cloud of the latest keyframe are marked as seen using fast voxel traversal. 
6. Repeat step 5 until all keyframes are processed.
### Outputs
The module's main focus is to visualize the voxels based on their status. The status have 8 different states:
___
## Implementation details
This module uses both **Open3D** library and two custom classes `Voxel` and `VoxelGrid`:

### `Voxel` class
- This class represents a single voxel.

Explanation of `Voxel`'s attributes:
* `x`, `y`, and `z` store the voxel's position as indices
* the `size` is the voxel's size i.e., height, width, depth
* the `color` stores the voxels color that is used for visualization purposes
* the `status` stores the category in which the voxel belongs to
* the `queued` store if the voxel has already been processed in the neighbours search while marking the exterior voxels

### `VoxelGrid` class
- This class represents a voxel grid.

Explanation of `VoxelGrid`'s attributes:
* `cell_size` stores the size of the voxels' i.e., height, width, depth
* `origin` stores the voxel grid's origin in real world coordinates
* `width`, `height`, and `depth` store the voxel grid's real-life size
* `width_voxels`, `height_voxels`, and `depth_voxels` store the voxel grid's dimensions in voxels
* `voxels` stores the voxels as a dictionary of voxels `self.voxels: Dict[Voxel]`
* `interior_voxels`, `exterior_voxels`, `mesh_voxels`, `scanned_voxels`, `seen_voxels`, and `unknown_voxels` store the indices of the voxels as they're being marked

### Point cloud generation
The point cloud reconstruction is implemented in `map_database_point_cloud_reconstruction.py`. 

## Notes
This module works with iPhone RGB-D images