"""Observation builder for constructing per-agent observation vectors."""

from typing import Dict
import numpy as np
import mujoco

from humanoid_hug.utils.ids import IDCache
from humanoid_hug.utils.kinematics import (
    get_forward_vector,
    get_up_vector,
    rotate_to_local_frame,
    compute_facing_alignment,
    compute_up_alignment,
    get_root_linear_velocity,
    get_root_angular_velocity,
)


class ObservationBuilder:
    """Builds per-agent observation vectors for the humanoid hug environment."""

    def __init__(self, id_cache: IDCache, model: mujoco.MjModel):
        """Initialize the observation builder.

        Args:
            id_cache: Cached IDs from the MuJoCo model
            model: MuJoCo model instance
        """
        self.id_cache = id_cache
        self.model = model
        self.agents = ["h0", "h1"]

        # Calculate observation dimension
        self._compute_obs_dim()

    def _compute_obs_dim(self) -> None:
        """Compute the observation dimension for each agent."""
        # Proprioception:
        # - Joint positions (excluding root): nq_agent - 7 (freejoint)
        # - Joint velocities (excluding root): nv_agent - 6 (freejoint)
        # - Root orientation (quaternion): 4
        # - Root angular velocity: 3

        # Get joint counts for one agent
        nq_agent = len(self.id_cache.joint_qpos_idx["h0"])
        nv_agent = len(self.id_cache.joint_qvel_idx["h0"])

        proprio_dim = (nq_agent - 7) + (nv_agent - 6) + 4 + 3  # joints + root quat + root angvel

        # Partner relative features:
        # - Relative position of partner chest (in local frame): 3
        # - Relative position of partner pelvis (in local frame): 3
        # - Relative velocity of partner root (in local frame): 3
        # - Facing alignment (dot product): 1
        # - Up alignment (dot product): 1
        partner_dim = 3 + 3 + 3 + 1 + 1

        # Contact bits:
        # - Self left arm touching partner torso: 1
        # - Self right arm touching partner torso: 1
        # - Partner arm touching self torso: 1
        contact_dim = 3

        self.obs_dim = proprio_dim + partner_dim + contact_dim

    def get_obs_dim(self) -> int:
        """Get the observation dimension.

        Returns:
            Size of the observation vector
        """
        return self.obs_dim

    def build_observations(
        self,
        data: mujoco.MjData,
        contact_info: Dict[str, bool]
    ) -> Dict[str, np.ndarray]:
        """Build observation vectors for both agents.

        Args:
            data: MuJoCo data instance
            contact_info: Contact detection results from ContactDetector

        Returns:
            Dictionary mapping agent ID to observation array
        """
        obs = {}
        for agent in self.agents:
            obs[agent] = self._build_agent_obs(data, agent, contact_info)
        return obs

    def _build_agent_obs(
        self,
        data: mujoco.MjData,
        agent: str,
        contact_info: Dict[str, bool]
    ) -> np.ndarray:
        """Build observation vector for a single agent.

        Args:
            data: MuJoCo data instance
            agent: Agent ID ('h0' or 'h1')
            contact_info: Contact detection results

        Returns:
            Flat observation array
        """
        partner = "h1" if agent == "h0" else "h0"

        obs_parts = []

        # === Proprioception ===
        qpos_idx = self.id_cache.joint_qpos_idx[agent]
        qvel_idx = self.id_cache.joint_qvel_idx[agent]

        # Joint positions (excluding root position, keeping root quat)
        # Freejoint: [x,y,z, qw,qx,qy,qz, joint1, joint2, ...]
        joint_qpos = data.qpos[qpos_idx[7:]]  # Skip root pos (3) and quat (4)
        obs_parts.append(joint_qpos)

        # Joint velocities (excluding root linear velocity)
        # Freejoint: [vx,vy,vz, wx,wy,wz, joint1_vel, ...]
        joint_qvel = data.qvel[qvel_idx[6:]]  # Skip root lin (3) and ang (3) vel
        obs_parts.append(joint_qvel)

        # Root orientation (quaternion)
        root_quat = data.qpos[qpos_idx[3:7]]
        obs_parts.append(root_quat)

        # Root angular velocity
        root_angvel = get_root_angular_velocity(data, qvel_idx)
        obs_parts.append(root_angvel)

        # === Partner relative features ===
        # Get torso transforms
        self_xpos = self.id_cache.get_torso_xpos(data, agent)
        self_xmat = self.id_cache.get_torso_xmat(data, agent)
        partner_xpos = self.id_cache.get_torso_xpos(data, partner)
        partner_xmat = self.id_cache.get_torso_xmat(data, partner)

        # Relative position of partner chest (in local frame)
        try:
            partner_chest = self.id_cache.get_site_xpos(data, f"{partner}_chest")
        except ValueError:
            partner_chest = partner_xpos
        rel_chest = partner_chest - self_xpos
        rel_chest_local = rotate_to_local_frame(rel_chest, self_xmat)
        obs_parts.append(rel_chest_local)

        # Relative position of partner pelvis (in local frame)
        try:
            partner_pelvis = self.id_cache.get_site_xpos(data, f"{partner}_pelvis")
        except ValueError:
            partner_pelvis = partner_xpos - np.array([0, 0, 0.2])
        rel_pelvis = partner_pelvis - self_xpos
        rel_pelvis_local = rotate_to_local_frame(rel_pelvis, self_xmat)
        obs_parts.append(rel_pelvis_local)

        # Relative velocity of partner root (in local frame)
        partner_qvel_idx = self.id_cache.joint_qvel_idx[partner]
        partner_linvel = get_root_linear_velocity(data, partner_qvel_idx)
        self_linvel = get_root_linear_velocity(data, qvel_idx)
        rel_vel = partner_linvel - self_linvel
        rel_vel_local = rotate_to_local_frame(rel_vel, self_xmat)
        obs_parts.append(rel_vel_local)

        # Facing alignment
        self_fwd = get_forward_vector(self_xmat)
        partner_fwd = get_forward_vector(partner_xmat)
        facing = compute_facing_alignment(self_fwd, partner_fwd)
        obs_parts.append(np.array([facing]))

        # Up alignment
        self_up = get_up_vector(self_xmat)
        partner_up = get_up_vector(partner_xmat)
        up_align = compute_up_alignment(self_up, partner_up)
        obs_parts.append(np.array([up_align]))

        # === Contact bits ===
        if agent == "h0":
            l_arm_contact = float(contact_info.get("h0_l_arm_torso", False))
            r_arm_contact = float(contact_info.get("h0_r_arm_torso", False))
            partner_arm_contact = float(contact_info.get("h1_arm_torso", False))
        else:
            l_arm_contact = float(contact_info.get("h1_l_arm_torso", False))
            r_arm_contact = float(contact_info.get("h1_r_arm_torso", False))
            partner_arm_contact = float(contact_info.get("h0_arm_torso", False))

        obs_parts.append(np.array([l_arm_contact, r_arm_contact, partner_arm_contact]))

        # Concatenate all parts
        obs = np.concatenate(obs_parts).astype(np.float32)

        return obs
