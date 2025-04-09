"""
pose_keyframe.py

This module defines the Pose and Keyframe classes for representing 3D poses and keyframes.

Classes:
    - Pose: Represents a 3D pose (position and orientation).
    - Keyframe: Extends Pose and represents a keyframe with additional attributes.

Usage:
    # Example usage
    from pose_keyframe import Pose, Keyframe

    # Create a Pose instance
    pose_instance = Pose(x=1.0, y=2.0, z=0.5, qx=0.0, qy=0.0, qw=1.0)

    # Create a Keyframe from a Pose instance
    keyframe_from_pose = Keyframe(id=1, map_id=42, pose=pose_instance)

    # Alternatively, create a Keyframe directly with individual attributes
    keyframe_from_attributes = Keyframe(id=2, map_id=42, x=2.0, y=3.0, z=0.7, qx=0.1, qy=0.2, qw=0.9)
"""


class Pose:
    """
    Represents a 3D pose (position and orientation).

    Attributes:
        x (float): X-coordinate of the position.
        y (float): Y-coordinate of the position.
        z (float): Z-coordinate of the position.
        qx (float): Quaternion x-component (orientation).
        qy (float): Quaternion y-component (orientation).
        qw (float): Quaternion w-component (orientation).

    Example:
        pose = Pose(x=1.0, y=2.0, z=0.5, qx=0.0, qy=0.0, qw=1.0)
    """
    def __init__(self, x: float, y: float, z: float, qx: float, qy: float, qz:float, qw: float) -> None:
        """
        Initialize a Pose instance.

        Args:
            x (float): X-coordinate of the position.
            y (float): Y-coordinate of the position.
            z (float): Z-coordinate of the position.
            qx (float): Quaternion x-component (orientation).
            qy (float): Quaternion y-component (orientation).
            qz (float): Quaternion z-component (orientation).
            qw (float): Quaternion w-component (orientation).
        """
        self.x = x
        self.y = y
        self.z = z
        self.qx = qx
        self.qy = qy
        self.qz = qz
        self.qw = qw


class Keyframe(Pose):
    """
    Represents a keyframe with additional attributes.

    Attributes:
        Inherits attributes from Pose class.
        id (int): Unique identifier for the keyframe.
        map_id (int): Identifier of the associated map.
        x (float): X-coordinate of the position.
        y (float): Y-coordinate of the position.
        z (float): Z-coordinate of the position.
        q_x (float): Quaternion x-component (orientation).
        q_y (float): Quaternion y-component (orientation).
        q_z (float): Quaternion z-component (orientation).
        q_w (float): Quaternion w-component (orientation).


    Example:

        # Create a Keyframe directly with individual attributes
        keyframe_from_attributes = Keyframe(id=2, map_id=42, x=2.0, y=3.0, z=0.7, qx=0.1, qy=0.2, qw=0.9)
        
        # Alternatively, create a Keyframe from a Pose instance
        keyframe_from_pose = Keyframe(id=1, map_id=42, pose=pose_instance)
    """
    def __init__(self, id, map_id, x=None, y=None, z=None, q_x=None, q_y=None, q_z=None, q_w=None, pose=None) -> None:
        if pose:
            # Initialize from a Pose instance
            super().__init__(pose.x, pose.y, pose.z, pose.qx, pose.qy, pose.qw)
        else:
            # Initialize with individual attributes
            super().__init__(x, y, z, q_x, q_y, q_z, q_w)
        self.id = id
        self.map_id = map_id

    def __str__(self):
        return f"\n{'-----Keyframe Info-----':^20}\nID: {self.id} | MapID: {self.map_id}\n" \
            f"x: {self.x}\ny: {self.y}\nz: {self.z}\n" \
            f"qx: {self.qx}\nqy: {self.qy}\nqz: {self.qz}\nqw: {self.qw}"
