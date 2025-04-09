import sqlite3
from typing import List, Tuple
from src.cameraframe import CameraFrame
from src.pose_keyframe import Keyframe
import open3d as o3d
import numpy as np


class MapDatabasePointCloudReconstruction:

    def __init__(self, db_name: str) -> None:
        self.db_connection = sqlite3.connect(db_name)

    def close(self):
        self.db_connection.close()

    def create_cursor(self):
        return self.db_connection.cursor()

    def list_keyframes(self, map_id) -> List[Tuple[int, int]]:
        """Get the list of all keyframes\' keyframe_id and map_id"""
        cur = self.create_cursor()
        cur.execute(f"SELECT keyframe_id, map_id FROM Keyframe WHERE map_id = {map_id}")
        keyframes: List[Tuple[int, int]] = cur.fetchall()
        return keyframes

    def get_keyframe(self, keyframe: Tuple[int, int]) -> Keyframe:
        """
        Get the pose of a keyframe given keyframe's PrimaryKey (keyframe_id, map_id).
        """
        cur = self.create_cursor()
        result = cur.execute(f"SELECT x, y, z, q_x, q_y, q_z, q_w FROM Keyframe WHERE keyframe_id = ? AND map_id = ?",
                             (keyframe[0], keyframe[1])).fetchall()[0]
        pose = Keyframe(id=keyframe[0], map_id=keyframe[1],
                        x=result[0], y=result[1], z=result[2],
                        q_x=result[3], q_y=result[4], q_z=result[5], q_w=result[6])

        return pose

    def get_agents(self) -> List[Tuple[int, str]]:
        """Get the ids and names of the agents from Agent table"""
        cur = self.create_cursor()
        cur.execute("SELECT * from Agent")
        agents = cur.fetchall()
        return agents

    def get_local_maps(self) -> List[Tuple[int, int]]:
        """
        This function returns a list of tuples from the LocalMap table, where each tuple represents a local map.
        Each tuple has the structure: (map_id: int, agent_id: int)
        """
        cur = self.create_cursor()
        cur.execute("SELECT * FROM LocalMap")
        local_maps = cur.fetchall()
        return local_maps

    def get_intrinsics(self, camera_id: int, map_id: int):
        cur = self.create_cursor()
        return cur.execute("SELECT is_depth, fx, fy, cx, cy FROM Camera WHERE camera_id = ? AND map_id = ?",
                           (camera_id, map_id)).fetchall()[0]

    def generate_pcd(self, keyframe_id, map_id, voxel_size: float = None,
                     transformation_matrix: np.array = None) -> o3d.geometry.PointCloud:
        """
        Generates a pointcloud from a keyframe given the kayframe_id and the map_id.
        """
        cur = self.create_cursor()
        cur.execute("SELECT camera_id, image FROM Image WHERE keyframe_id = ? and map_id = ?", (keyframe_id, map_id))
        result = cur.fetchall()  # List[Tuple[camera_id, image_bytes]]

        assert len(result) == 2, 'One or more camera(s) retrieved for this keyframe.'

        intrinsic_0 = self.get_intrinsics(result[0][0], map_id)
        cameraFrame0 = CameraFrame(camera_id=result[0][0], is_depth=intrinsic_0[0],
                                fx=intrinsic_0[1], fy=intrinsic_0[2],
                                cx=intrinsic_0[3], cy=intrinsic_0[4],
                                image=result[0][1])

        intrinsic_1 = self.get_intrinsics(result[1][0], map_id)
        cameraFrame1 = CameraFrame(camera_id=result[1][0], is_depth=intrinsic_1[0],
                                fx=intrinsic_1[1], fy=intrinsic_1[2],
                                cx=intrinsic_1[3], cy=intrinsic_1[4],
                                image=result[1][1])
        # cameraFrame0.display()
        # cameraFrame1.display()

        assert cameraFrame0.is_depth != cameraFrame1.is_depth, 'Both images are RGB or both are depth.'
        intrinsic_cam_parameters = o3d.camera.PinholeCameraIntrinsic()
        intrinsic_cam_parameters.set_intrinsics(cameraFrame0.width, cameraFrame0.height, cameraFrame0.fx, cameraFrame0.fy,
                                                cameraFrame0.cx, cameraFrame0.cy)

        if cameraFrame0.is_depth == 1 and cameraFrame1.is_depth == 0:
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image((cameraFrame1.img)),
                                                                      o3d.geometry.Image((cameraFrame0.img)),
                                                                      depth_scale=50 / 255, depth_trunc=5,
                                                                      convert_rgb_to_intensity=False)
        else:

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image((cameraFrame0.img)),
                                                                      o3d.geometry.Image((cameraFrame1.img)),
                                                                      depth_scale=50 / 255, depth_trunc=5,
                                                                      convert_rgb_to_intensity=False)

        pose = self.get_keyframe(keyframe=(keyframe_id, map_id))

        translation_vector = np.array([pose.x, pose.y, pose.z])

        rotation_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(
            np.asarray([pose.qw, pose.qx, pose.qy, pose.qz]))

        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = rotation_matrix
        extrinsic_matrix[:3, 3] = translation_vector.T

        pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic_cam_parameters)

        # downsample the point cloud
        if voxel_size is not None:
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

        pcd.transform(extrinsic_matrix)

        if transformation_matrix is not None:
            pcd.transform(transformation_matrix)

        pcd.estimate_normals()
        # pcd.orient_normals_towards_camera_location(camera_location=translation_vector)

        return pcd
