import numpy as np
import open3d as o3d
from typing import Dict, Tuple, List
from queue import Queue
from enum import Enum


class Status(Enum):
    Mesh = 1
    Debug = 2
    Selected = 3
    Exterior = 4
    Scanned = 5
    Seen = 6
    Unknown = 7
    Temporary = 8

    @property
    def rgb(self):
        color_map = {
            1: np.array([1, 0, 0]),  # red
            2: np.array([0, 1, 0]),  # green
            3: np.array([0, 0, 1]),  # blue
            4: np.array([1, 1, 0]),  # yellow
            5: np.array([0, 1, 1]),  # cyan
            6: np.array([0.902, 0.702, 1]),  # magenta
            7: np.array([0.337, 0.788, 0.533]),  # green blue
            8: np.array([0.5, 0.5, 0.5])  # grey
            }
        return color_map[self.value]


class Voxel:
    def __init__(self, x: int, y: int, z: int, size: float, color: np.array = np.array([0.8, 0.6, 1])):
        self.x = x
        self.y = y
        self.z = z
        self.size = size
        self.color = Status.Unknown.rgb
        self.status = Status.Unknown
        self.queued = 0

    def set_status(self, status: Status, color = None):
        self.status = status
        if status == Status.Scanned:
            self.color = color
        else:
            self.color = status.rgb

    def set_queued(self):
        self.queued = 1

    def to_world_coords(self, cell_size: float, origin: np.array):
        return np.array([self.x, self.y, self.z]) * cell_size + origin

    def to_world_coords2(self, cell_size: float, origin: np.array):
        # return the min and max coordinates of the voxel in world coordinates
        return np.array([self.x, self.y, self.z]) * cell_size + origin, np.array([self.x + self.size, self.y + self.size, self.z + self.size]) * cell_size + origin


class VoxelGrid:
    """
    VoxelGrid class for representing a 3D grid of voxels.
    """
    color_attr: str = np.array([1, 0, 0])

    def __init__(self,
                 cell_size: float = 1.,
                 origin: np.array = np.array([0, 0, 0]),
                 min_bound: np.array = None,
                 max_bound: np.array = None,
                 # color_attr: np.array = None,
                 voxels: list = None
                 ):

        self.cell_size: float = cell_size
        self.origin: np.array = origin
        self.width = int(np.ceil(max_bound[0])) - int(np.floor(min_bound[0]))
        self.height = int(np.ceil(max_bound[1])) - int(np.floor(min_bound[1]))
        self.depth = int(np.ceil(max_bound[2])) - int(np.floor(min_bound[2]))
        self.width_voxels = int(np.ceil(self.width / self.cell_size))
        self.height_voxels = int(np.ceil(self.height / self.cell_size))
        self.depth_voxels = int(np.ceil(self.depth / self.cell_size))
        # self.color = color_attr
        self.voxels: Dict[Voxel] = {}
        # self.interior_voxels = []
        self.exterior_voxels = []
        self.mesh_voxels = []
        self.scanned_voxels = []
        self.seen_voxels = []
        self.unknown_voxels = []

        if voxels is not None and len(voxels) > 0:
            # Create voxels from o3d.geometry.VoxelGrid
            for voxel in voxels:
                self.voxels[tuple(voxel.grid_index)] = Voxel(voxel.grid_index[0],
                                                             voxel.grid_index[1],
                                                             voxel.grid_index[2],
                                                             cell_size)
                self.voxels[tuple(voxel.grid_index)].set_status(Status.Mesh)
            self.mesh_voxels = [tuple(voxel.grid_index) for voxel in voxels]

        else:
            for i in range(int(np.ceil(self.width / self.cell_size))):
                for j in range(int(np.ceil(self.height / self.cell_size))):
                    for k in range(int(np.ceil(self.depth / self.cell_size))):
                        voxel = (i, j, k)
                        self.voxels[voxel] = Voxel(i, j, k, cell_size)
        self.size = len(self.voxels) if self.voxels else 0

    def fast_voxel_traversal(self, start: np.array, end: np.array) -> list[tuple]:
        visited = []
        # Convert start and end points to grid coordinates
        start_voxel = self.to_grid_coords(start)
        end_voxel = self.to_grid_coords(end)

        # Check if start voxel is outside the grid
        if (start_voxel < 0).any() or (start_voxel >= [self.width_voxels, self.height_voxels, self.depth_voxels]).any():
            # Create a temporary voxel for traversal
            temp_start_voxel = Voxel(start_voxel[0], start_voxel[1], start_voxel[2], self.cell_size)
            temp_start_voxel.set_status(Status.Temporary)
            self.voxels[tuple(start_voxel)] = temp_start_voxel
        else:
            start_voxel = np.clip(start_voxel, [0, 0, 0],
                                  [self.width_voxels - 1, self.height_voxels - 1, self.depth_voxels - 1])

        # Initialize current voxel to start voxel
        current_voxel = start_voxel.copy()

        # Calculate the direction vector
        direction = end - start
        direction = direction / np.linalg.norm(direction)

        # Calculate step direction
        step = np.sign(direction).astype(int)

        # Calculate tMax and tDelta
        tMax = np.zeros(3)
        tDelta = np.zeros(3)
        for i in range(3):
            if direction[i] != 0:
                tMax[i] = ((current_voxel[i] + (step[i] > 0)) * self.cell_size + self.origin[i] - start[i]) / direction[i]
                tDelta[i] = self.cell_size / abs(direction[i])
            else:
                tMax[i] = np.inf
                tDelta[i] = np.inf

        # Traverse the grid
        while not np.array_equal(current_voxel, end_voxel):
            # Find the axis with the smallest tMax
            axis = np.argmin(tMax)

            # Move to the next voxel in the direction of the smallest tMax
            current_voxel[axis] += step[axis]
            tMax[axis] += tDelta[axis]

            # Check if the current voxel is within bounds
            if (current_voxel < 0).any() or (current_voxel >= [self.width_voxels, self.height_voxels, self.depth_voxels]).any():
                continue

            # Mark the voxel as seen
            elif self.voxels[tuple(current_voxel)].status == Status.Unknown:
                visited.append(tuple(current_voxel))
                # self.voxels[tuple(current_voxel)].set_status(Status.Seen)
                self.seen_voxels.append(tuple(current_voxel))
        if len(visited) > 0:
            return visited

    def mark_scanned_voxels(self, voxels: o3d.geometry.VoxelGrid, pose: np.array) -> None:
        origin = voxels.origin
        cell_size = voxels.voxel_size
        for voxel in voxels.get_voxels():
            world_coords = np.array(voxel.grid_index) * cell_size + origin
            grid_coords = np.floor((world_coords - self.origin) / self.cell_size).astype(int)
            if grid_coords[0] >= self.width_voxels or grid_coords[0] < 0 or grid_coords[1] >= self.height_voxels or grid_coords[1] < 0 or grid_coords[2] >= self.depth_voxels or grid_coords[2] < 0:
                continue
            else:
                self.voxels[tuple(voxel.grid_index)].set_status(Status.Scanned, voxel.color)
                self.scanned_voxels.append(tuple(voxel.grid_index))

    def mark_seen(self, voxels: o3d.geometry.VoxelGrid, pose: np.array) -> None:
        origin = voxels.origin
        cell_size = voxels.voxel_size
        for voxel in voxels.get_voxels():
            world_coords = np.array(voxel.grid_index) * cell_size + origin
            grid_coords = np.floor((world_coords - self.origin) / self.cell_size).astype(int)
            if grid_coords[0] >= self.width_voxels or grid_coords[0] < 0 or grid_coords[1] >= self.height_voxels or grid_coords[1] < 0 or grid_coords[2] >= self.depth_voxels or grid_coords[2] < 0:
                continue
            else:
                visited_voxels = self.fast_voxel_traversal(np.array([pose[0][0], pose[0][1], pose[0][2]]), world_coords)
                if visited_voxels:
                    for vv in visited_voxels:
                        self.voxels[vv].set_status(Status.Seen)
                        self.seen_voxels.append(tuple(voxel.grid_index))

    def make_dense(self) -> None:
        for i in range(int(np.ceil(self.width / self.cell_size))):
            for j in range(int(np.ceil(self.height / self.cell_size))):
                for k in range(int(np.ceil(self.depth / self.cell_size))):
                    key = (i, j, k)
                    if key not in self.voxels:
                        self.voxels[key] = Voxel(i, j, k, self.cell_size, color=np.array([0, 0, 0]))
        self.size = len(self.voxels)

    def find_6_neighbors(self, voxel) -> list[Voxel]:
        # check that voxel is in the grid
        try:
            self.voxels[voxel]
        except KeyError:
            print("Voxel is out of bounds, neighbors cannot be found.")
            return []

        if voxel[0] == 0:
            range_x = range(1, 2)
        elif voxel[0] == self.width - 1:
            range_x = range(-1, 0)
        else:
            range_x = range(-1, 2)

        if voxel[1] == 0:
            range_y = range(1, 2)
        elif voxel[1] == self.height - 1:
            range_y = range(-1, 0)
        else:
            range_y = range(-1, 2)

        if voxel[2] == 0:
            range_z = range(1, 2)
        elif voxel[2] == self.depth - 1:
            range_z = range(-1, 0)
        else:
            range_z = range(-1, 2)

        neighbors = []
        for i in range_x:
            if i != 0:
                neighbors.append(self.voxels[(voxel[0] + i, voxel[1], voxel[2])])

        for j in range_y:
            if j != 0:
                neighbors.append(self.voxels[(voxel[0], voxel[1] + j, voxel[2])])

        for k in range_z:
            if k != 0:
                neighbors.append(self.voxels[(voxel[0], voxel[1], voxel[2] + k)])

        return neighbors

    def mark_exterior_voxels(self) -> None:
        # Create a queue of voxels to visit
        q = Queue()
        if self.voxels[(0, 0, 0)].status == Status.Unknown:
            q.put(self.voxels[(0, 0, 0)])
            self.voxels[(0, 0, 0)].set_queued()
        elif self.voxels[(self.width_voxels - 1, self.height_voxels - 1, self.depth_voxels - 1)].status == Status.Unknown:
            q.put(self.voxels[(self.width_voxels - 1, self.height_voxels - 1, self.depth_voxels - 1)])
            self.voxels[(self.width_voxels - 1, self.height_voxels - 1, self.depth_voxels - 1)].set_queued()

        try:
            while not q.empty():

                voxel = q.get()
                voxel.set_status(Status.Exterior)
                self.exterior_voxels.append((voxel.x, voxel.y, voxel.z))
                # Create ranges based on voxel
                if voxel.x == 0:
                    range_x = range(1, 2)
                elif voxel.x == self.width_voxels - 1:
                    range_x = range(-1, 0)
                else:
                    range_x = range(-1, 2)

                if voxel.y == 0:
                    range_y = range(1, 2)
                elif voxel.y == self.height_voxels - 1:
                    range_y = range(-1, 0)
                else:
                    range_y = range(-1, 2)

                if voxel.z == 0:
                    range_z = range(1, 2)
                elif voxel.z == self.depth_voxels - 1:
                    range_z = range(-1, 0)
                else:
                    range_z = range(-1, 2)

                for i in range_x:
                    if i != 0:
                        neighbor = self.voxels[(voxel.x + i, voxel.y, voxel.z)]
                        if neighbor.status == Status.Unknown and neighbor.queued == 0:
                            neighbor.set_queued()
                            q.put(neighbor)

                for j in range_y:
                    if j != 0:
                        neighbor = self.voxels[(voxel.x, voxel.y + j, voxel.z)]
                        if neighbor.status == Status.Unknown and neighbor.queued == 0:
                            neighbor.set_queued()
                            q.put(neighbor)

                for k in range_z:
                    if k != 0:
                        neighbor = self.voxels[(voxel.x, voxel.y, voxel.z + k)]
                        if neighbor.status == Status.Unknown and neighbor.queued == 0:
                            neighbor.set_queued()
                            q.put(neighbor)
                # if Queue.qsize(q) > 2200:
                #
                #     for idx in Queue.qsize(q):
                #         voxel = q.get()
                #         voxel.set_status(Status.Exterior)
                #     break
        except:
            print(voxel.x,voxel.y,voxel.z)

    def to_world_coords(self, grid_coords) -> np.array:
        return np.array(grid_coords) * self.cell_size + self.origin

    def to_grid_coords(self, world_coords) -> np.array:
        return np.floor((world_coords - self.origin) / self.cell_size).astype(int)

    def contains_points_test(self, pcd: o3d.geometry.PointCloud):
        points = np.asarray(pcd.points)
        for i in range(self.width_voxels):
            for j in range(self.height_voxels):
                for k in range(self.depth_voxels):
                    voxel = self.voxels[(i, j, k)]
                    min_coords, max_coords = voxel.to_world_coords2(self.cell_size, self.origin)
                    if np.all(np.logical_and(points >= min_coords, points <= max_coords), axis=1).any():
                        voxel.set_status('Occupied')

    def rectify_voxels(self) -> None:
        for voxel in self.voxels:
            counter = 0
            if self.voxels[voxel].status == Status.Unknown:
                neighbors =  self.find_6_neighbors(voxel)
                for neighbor in neighbors:
                    if neighbor.status == Status.Unknown:
                        counter += 1
                    if counter <= 1:
                        self.voxels[voxel].set_status(Status.Seen)
                        self.seen_voxels.append(voxel)

    def finalize(self):
        # populate the unknown voxels
        for voxel in self.voxels:
            if self.voxels[voxel].status == Status.Unknown:
                self.unknown_voxels.append(voxel)

    def bbox_to_grid_coords(self, min_bound, max_bound):
        return (np.floor((min_bound - self.origin) / self.cell_size).astype(int), np.ceil((max_bound - self.origin) / self.cell_size).astype(int))

    def to_pcd(self, classification: list = None) -> o3d.geometry.PointCloud:
        if classification is None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(
                [self.to_world_coords(voxel) for voxel in self.voxels])
            pcd.colors = o3d.utility.Vector3dVector(
                [self.voxels[voxel].color for voxel in self.voxels])
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(
                [self.to_world_coords(voxel) for voxel in self.voxels if self.voxels[voxel].status in classification])
            pcd.colors = o3d.utility.Vector3dVector(
                [self.voxels[voxel].color for voxel in self.voxels if self.voxels[voxel].status in classification])
        return pcd

    def to_obj_points(self) -> None:
        with open('voxels.obj', 'w') as f:
            for voxel in self.voxels:
                if self.voxels[voxel].status == 'Occupied':
                    f.write(f'v {voxel[0]} {voxel[1]} {voxel[2]} 1 0 0\n')
                else:
                    f.write(f'v {voxel[0]} {voxel[1]} {voxel[2]} 0 1 1\n')

    def to_ply_faces(self, classification: Status | List[Status], store: bool = False) -> o3d.geometry.TriangleMesh:
        if isinstance(classification, Status):

            mesh = o3d.geometry.TriangleMesh()

            if classification == Status.Scanned:
                for voxel in getattr(self, f"{classification.name.lower()}_voxels"):
                    cube = o3d.geometry.TriangleMesh.create_box(width=self.cell_size,
                                                                height=self.cell_size,
                                                                depth=self.cell_size).translate(
                        np.array([voxel[0], voxel[1], voxel[2]])*self.cell_size + self.origin)

                    cube.vertex_colors = o3d.utility.Vector3dVector([self.voxels[voxel].color for _ in range(len(cube.vertices))])
                    mesh += cube
            else:
                for voxel in getattr(self, f"{classification.name.lower()}_voxels"):
                    cube = o3d.geometry.TriangleMesh.create_box(width=self.cell_size,
                                                                height=self.cell_size,
                                                                depth=self.cell_size).translate(
                        np.array([voxel[0], voxel[1], voxel[2]]) * self.cell_size + self.origin)
                    mesh += cube
                mesh.paint_uniform_color(classification.rgb)

            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()
            mesh.remove_duplicated_vertices()
            mesh.remove_duplicated_triangles()

            if store:
                o3d.io.write_triangle_mesh(f"data/{classification.name}.ply", mesh)
            return mesh

    def create_voxels(self, classification: list = None) -> o3d.geometry.VoxelGrid:
        if classification is None:
            voxels = o3d.geometry.VoxelGrid.create_from_point_cloud(self.to_pcd())
        else:
            voxels = o3d.geometry.VoxelGrid.create_from_point_cloud(self.to_pcd(classification), voxel_size=self.cell_size)
        return voxels
