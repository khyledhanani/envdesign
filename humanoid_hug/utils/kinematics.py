"""Kinematics utilities for coordinate transformations and body orientation."""

import numpy as np
import mujoco


def get_body_xmat(data: mujoco.MjData, body_id: int) -> np.ndarray:
    """Get the 3x3 rotation matrix of a body.

    Args:
        data: MuJoCo data instance
        body_id: Body ID

    Returns:
        3x3 rotation matrix
    """
    return data.xmat[body_id].reshape(3, 3)


def get_forward_vector(xmat: np.ndarray) -> np.ndarray:
    """Get the forward direction vector (x-axis) from a rotation matrix.

    In MuJoCo, the forward direction is typically the x-axis of the body frame.

    Args:
        xmat: 3x3 rotation matrix

    Returns:
        3D unit forward vector
    """
    forward = xmat[:, 0]
    return forward / (np.linalg.norm(forward) + 1e-8)


def get_up_vector(xmat: np.ndarray) -> np.ndarray:
    """Get the up direction vector (z-axis) from a rotation matrix.

    Args:
        xmat: 3x3 rotation matrix

    Returns:
        3D unit up vector
    """
    up = xmat[:, 2]
    return up / (np.linalg.norm(up) + 1e-8)


def get_right_vector(xmat: np.ndarray) -> np.ndarray:
    """Get the right direction vector (y-axis) from a rotation matrix.

    Args:
        xmat: 3x3 rotation matrix

    Returns:
        3D unit right vector
    """
    right = xmat[:, 1]
    return right / (np.linalg.norm(right) + 1e-8)


def rotate_to_local_frame(vec: np.ndarray, xmat: np.ndarray) -> np.ndarray:
    """Rotate a world-frame vector into a local body frame.

    Args:
        vec: 3D vector in world frame
        xmat: 3x3 rotation matrix of the body

    Returns:
        3D vector in local frame
    """
    return xmat.T @ vec


def rotate_to_world_frame(vec: np.ndarray, xmat: np.ndarray) -> np.ndarray:
    """Rotate a local body frame vector into world frame.

    Args:
        vec: 3D vector in local frame
        xmat: 3x3 rotation matrix of the body

    Returns:
        3D vector in world frame
    """
    return xmat @ vec


def quat_to_euler(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion to Euler angles (roll, pitch, yaw).

    Args:
        quat: Quaternion [w, x, y, z]

    Returns:
        Euler angles [roll, pitch, yaw] in radians
    """
    w, x, y, z = quat

    # Roll (rotation around x-axis)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (rotation around y-axis)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1, 1))

    # Yaw (rotation around z-axis)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


def compute_facing_alignment(fwd1: np.ndarray, fwd2: np.ndarray) -> float:
    """Compute facing alignment between two agents.

    For a hug, agents should face each other, meaning fwd1 should align
    with -fwd2.

    Args:
        fwd1: Forward vector of agent 1
        fwd2: Forward vector of agent 2

    Returns:
        Dot product of fwd1 and -fwd2 (higher = more aligned for hug)
    """
    return float(np.dot(fwd1, -fwd2))


def compute_up_alignment(up1: np.ndarray, up2: np.ndarray) -> float:
    """Compute up vector alignment between two agents.

    For a stable hug, both agents should be upright with similar up vectors.

    Args:
        up1: Up vector of agent 1
        up2: Up vector of agent 2

    Returns:
        Dot product of up vectors (higher = more aligned)
    """
    return float(np.dot(up1, up2))


def compute_tilt_angle(up_vec: np.ndarray) -> float:
    """Compute the tilt angle of a body from vertical.

    Args:
        up_vec: Up vector of the body

    Returns:
        Tilt angle in radians (0 = perfectly upright)
    """
    world_up = np.array([0.0, 0.0, 1.0])
    cos_angle = np.clip(np.dot(up_vec, world_up), -1, 1)
    return np.arccos(cos_angle)


def compute_relative_velocity(vel1: np.ndarray, vel2: np.ndarray) -> np.ndarray:
    """Compute relative velocity between two agents.

    Args:
        vel1: Velocity of agent 1
        vel2: Velocity of agent 2

    Returns:
        Relative velocity (vel1 - vel2)
    """
    return vel1 - vel2


def get_root_quat(data: mujoco.MjData, qpos_idx: np.ndarray) -> np.ndarray:
    """Extract root quaternion from qpos for a freejoint humanoid.

    The freejoint stores: [x, y, z, qw, qx, qy, qz, ...]

    Args:
        data: MuJoCo data instance
        qpos_idx: Array of qpos indices for the agent

    Returns:
        Quaternion [w, x, y, z]
    """
    # Root position is first 3, then quaternion is next 4
    return data.qpos[qpos_idx[3:7]].copy()


def get_root_position(data: mujoco.MjData, qpos_idx: np.ndarray) -> np.ndarray:
    """Extract root position from qpos for a freejoint humanoid.

    Args:
        data: MuJoCo data instance
        qpos_idx: Array of qpos indices for the agent

    Returns:
        Position [x, y, z]
    """
    return data.qpos[qpos_idx[0:3]].copy()


def get_root_linear_velocity(data: mujoco.MjData, qvel_idx: np.ndarray) -> np.ndarray:
    """Extract root linear velocity from qvel.

    The freejoint stores: [vx, vy, vz, wx, wy, wz, ...]

    Args:
        data: MuJoCo data instance
        qvel_idx: Array of qvel indices for the agent

    Returns:
        Linear velocity [vx, vy, vz]
    """
    return data.qvel[qvel_idx[0:3]].copy()


def get_root_angular_velocity(data: mujoco.MjData, qvel_idx: np.ndarray) -> np.ndarray:
    """Extract root angular velocity from qvel.

    Args:
        data: MuJoCo data instance
        qvel_idx: Array of qvel indices for the agent

    Returns:
        Angular velocity [wx, wy, wz]
    """
    return data.qvel[qvel_idx[3:6]].copy()
