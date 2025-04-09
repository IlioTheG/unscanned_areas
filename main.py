import numpy as np
from src.voxel_grid import VoxelGrid, Status
from src.map_database_point_cloud_reconstruction import MapDatabasePointCloudReconstruction
from tqdm import tqdm
from src.utils.material_generator import *


cell_size = 0.1
# Load Mesh
mesh = o3d.io.read_triangle_mesh("data/hull_new.ply")
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=mesh.get_min_bound())

# Create an o3d voxel grid from mesh
voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=cell_size)

# Create a custom voxel grid from the o3d voxel grid
voxel_grid_mine = VoxelGrid(cell_size=cell_size, origin=voxel_grid.origin, voxels=voxel_grid.get_voxels(), min_bound=mesh.get_min_bound(), max_bound=mesh.get_max_bound())
voxel_grid_mine.make_dense()
voxel_grid_mine.mark_exterior_voxels()


pcd = o3d.geometry.PointCloud()
camera_pcd = o3d.geometry.PointCloud()  # for visualization purposes
camera_line = o3d.geometry.LineSet()  # for visualization purposes

transformation_matrix = np.array([[0.965925812721, 0.000000000000, -0.258819043636, 0.957580983639],
                                 [0.000000000000, 1.000000000000, 0.000000000000, 0.000000000000],
                                 [0.258819043636, 0.000000000000, 0.965925812721, 1.222774863243],
                                 [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]])

db = MapDatabasePointCloudReconstruction(db_name="data/scan.db")

localmaps = db.get_local_maps()
print(localmaps)
for map_id, agent_id in localmaps:
    keyframes = db.list_keyframes(map_id)
    for keyframe in tqdm(keyframes[:], desc=f"Reconstructing map & marking voxels", colour='green', unit="keyframes"):
        # Get the pose and transform the x, y, z based on the transformation matrix for visualization purposes
        pose = db.get_keyframe(keyframe)
        camera_temp = o3d.geometry.PointCloud()
        camera_temp.points.append([pose.x, pose.y, pose.z])
        camera_temp.transform(transformation_matrix)
        camera_pcd.points.extend(camera_temp.points)
        camera_pcd.points.append([np.asarray(camera_temp.points)[0][0], np.asarray(camera_temp.points)[0][1], np.asarray(camera_temp.points)[0][2]])

        # Get the pcd and transform it based on the transformation matrix
        pcd_temp = db.generate_pcd(keyframe[0], keyframe[1], voxel_size=cell_size) # scan point cloud
        pcd_temp.transform(transformation_matrix)

        # Voxelize temporary point cloud with o3d
        scanned_voxels = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(pcd_temp, voxel_size=cell_size,
                                                                                      min_bound=mesh.get_min_bound(),
                                                                                      max_bound=mesh.get_max_bound())

        pcd.points.extend(pcd_temp.points)
        pcd.colors.extend(pcd_temp.colors)

        # Update the voxel grid with the scanned voxels
        voxel_grid_mine.mark_scanned_voxels(scanned_voxels, np.asarray(camera_temp.points))
        voxel_grid_mine.mark_seen(scanned_voxels, np.asarray(camera_temp.points))

db.close()

voxel_grid_mine.finalize()
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=60))
# update camera line using Vector2iVector
camera_line.points = camera_pcd.points
camera_line.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(camera_pcd.points)-1)])
camera_line.paint_uniform_color([0, 0, 1])
print(type(camera_pcd))
# Visualize the voxel grid
occupied = voxel_grid_mine.to_ply_faces(Status.Mesh, store=True)
scanned = voxel_grid_mine.to_ply_faces(Status.Scanned, store=True)
seen = voxel_grid_mine.to_ply_faces(Status.Seen, store=True)
unknown = voxel_grid_mine.to_ply_faces(Status.Unknown, store=True)
exterior = voxel_grid_mine.to_ply_faces(Status.Exterior, store=True)
# temporary = voxel_grid_mine.to_ply_faces(Status.Temporary, store=True)
o3d.io.write_point_cloud('pointcloud.ply', pointcloud=pcd, print_progress=True)
o3d.io.write_line_set(filename='camera_line.ply', line_set=camera_line)
# o3d.io.write_point_cloud(filename='camera_pcd.ply', poitncloud=camera_pcd)
o3d.visualization.draw([
    {'name': 'Mesh', 'geometry': mesh, 'material': general_mat},
    {'name': 'Exterior voxels', 'geometry': exterior, 'material': general_mat},
    {'name': 'Occupied voxels', 'geometry': occupied, 'material': occupied_mat},
    {'name': 'Scanned voxels', 'geometry': scanned, 'material': general_mat},
    {'name': 'Seen voxels', 'geometry': seen, 'material': seen_mat},
    {'name': 'Unknown voxels', 'geometry': unknown, 'material': unknown_mat},
    {'name': 'Camera', 'geometry': camera_pcd, 'material': general_mat},
    {'name': 'Camera Line', 'geometry': camera_line, 'material': general_mat},
    {'name': 'pcd', 'geometry': pcd}

],
    show_skybox=False, show_ui=True, bg_color=[0.411, 0.454, 0.454, 0.525], ibl_intensity=100000)
