"""ID caching utilities for efficient MuJoCo lookups."""

from typing import Dict, Set, List
import numpy as np
import mujoco


class IDCache:
    """Caches MuJoCo IDs for sites, geoms, bodies, joints, and actuators."""

    def __init__(self, model: mujoco.MjModel):
        """Initialize the ID cache from a MuJoCo model.

        Args:
            model: MuJoCo model instance
        """
        self.model = model
        self.agents = ["h0", "h1"]

        # Site IDs
        self.site_ids: Dict[str, int] = {}

        # Geom ID sets for contact detection
        self.arm_geoms: Dict[str, Set[int]] = {"h0": set(), "h1": set()}
        self.torso_geoms: Dict[str, Set[int]] = {"h0": set(), "h1": set()}
        self.left_arm_geoms: Dict[str, Set[int]] = {"h0": set(), "h1": set()}
        self.right_arm_geoms: Dict[str, Set[int]] = {"h0": set(), "h1": set()}

        # Body IDs
        self.body_ids: Dict[str, int] = {}
        self.torso_body_ids: Dict[str, int] = {}

        # Actuator indices per agent
        self.actuator_idx: Dict[str, np.ndarray] = {}

        # Joint indices per agent (for qpos/qvel slicing)
        self.joint_qpos_idx: Dict[str, np.ndarray] = {}
        self.joint_qvel_idx: Dict[str, np.ndarray] = {}

        self._cache_all()

    def _cache_all(self) -> None:
        """Cache all relevant IDs from the model."""
        self._cache_sites()
        self._cache_geoms()
        self._cache_bodies()
        self._cache_actuators()
        self._cache_joints()

    def _cache_sites(self) -> None:
        """Cache site IDs for both humanoids."""
        site_names = [
            "chest", "pelvis", "back", "head_site", "lhand", "rhand"
        ]
        for agent in self.agents:
            for name in site_names:
                full_name = f"{agent}_{name}"
                try:
                    site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, full_name)
                    if site_id >= 0:
                        self.site_ids[full_name] = site_id
                except Exception:
                    pass

    def _cache_geoms(self) -> None:
        """Cache geom IDs for arms and torso of each humanoid."""
        arm_keywords = ["upper_arm", "lower_arm", "hand"]
        left_arm_keywords = ["left_upper_arm", "left_lower_arm", "left_hand"]
        right_arm_keywords = ["right_upper_arm", "right_lower_arm", "right_hand"]
        torso_keywords = ["torso_geom", "chest_geom", "abdomen_geom"]

        for i in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if geom_name is None:
                continue

            for agent in self.agents:
                if not geom_name.startswith(f"{agent}_"):
                    continue

                # Check for arm geoms
                for kw in arm_keywords:
                    if kw in geom_name:
                        self.arm_geoms[agent].add(i)
                        break

                # Check for left arm geoms
                for kw in left_arm_keywords:
                    if kw in geom_name:
                        self.left_arm_geoms[agent].add(i)
                        break

                # Check for right arm geoms
                for kw in right_arm_keywords:
                    if kw in geom_name:
                        self.right_arm_geoms[agent].add(i)
                        break

                # Check for torso geoms
                for kw in torso_keywords:
                    if kw in geom_name:
                        self.torso_geoms[agent].add(i)
                        break

    def _cache_bodies(self) -> None:
        """Cache body IDs for both humanoids."""
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name is None:
                continue

            for agent in self.agents:
                if body_name.startswith(f"{agent}_"):
                    self.body_ids[body_name] = i
                    if body_name == f"{agent}_torso":
                        self.torso_body_ids[agent] = i

    def _cache_actuators(self) -> None:
        """Cache actuator indices for each humanoid."""
        for agent in self.agents:
            indices = []
            for i in range(self.model.nu):
                act_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                if act_name is not None and act_name.startswith(f"{agent}_"):
                    indices.append(i)
            self.actuator_idx[agent] = np.array(indices, dtype=np.int32)

    def _cache_joints(self) -> None:
        """Cache joint qpos/qvel indices for each humanoid."""
        for agent in self.agents:
            qpos_indices = []
            qvel_indices = []

            for i in range(self.model.njnt):
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if joint_name is None:
                    continue

                if joint_name.startswith(f"{agent}_"):
                    # Get qpos range for this joint
                    qpos_start = self.model.jnt_qposadr[i]
                    qvel_start = self.model.jnt_dofadr[i]

                    joint_type = self.model.jnt_type[i]

                    # Determine number of qpos/qvel elements
                    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                        # Free joint: 7 qpos (pos + quat), 6 qvel
                        qpos_indices.extend(range(qpos_start, qpos_start + 7))
                        qvel_indices.extend(range(qvel_start, qvel_start + 6))
                    elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
                        # Ball joint: 4 qpos (quat), 3 qvel
                        qpos_indices.extend(range(qpos_start, qpos_start + 4))
                        qvel_indices.extend(range(qvel_start, qvel_start + 3))
                    else:
                        # Hinge or slide: 1 qpos, 1 qvel
                        qpos_indices.append(qpos_start)
                        qvel_indices.append(qvel_start)

            self.joint_qpos_idx[agent] = np.array(qpos_indices, dtype=np.int32)
            self.joint_qvel_idx[agent] = np.array(qvel_indices, dtype=np.int32)

    def get_site_xpos(self, data: mujoco.MjData, site_name: str) -> np.ndarray:
        """Get the position of a site.

        Args:
            data: MuJoCo data instance
            site_name: Name of the site (e.g., 'h0_chest')

        Returns:
            3D position array
        """
        site_id = self.site_ids.get(site_name)
        if site_id is None:
            raise ValueError(f"Site '{site_name}' not found in cache")
        return data.site_xpos[site_id].copy()

    def get_body_xpos(self, data: mujoco.MjData, body_name: str) -> np.ndarray:
        """Get the position of a body.

        Args:
            data: MuJoCo data instance
            body_name: Name of the body

        Returns:
            3D position array
        """
        body_id = self.body_ids.get(body_name)
        if body_id is None:
            raise ValueError(f"Body '{body_name}' not found in cache")
        return data.xpos[body_id].copy()

    def get_body_xmat(self, data: mujoco.MjData, body_name: str) -> np.ndarray:
        """Get the rotation matrix of a body.

        Args:
            data: MuJoCo data instance
            body_name: Name of the body

        Returns:
            3x3 rotation matrix
        """
        body_id = self.body_ids.get(body_name)
        if body_id is None:
            raise ValueError(f"Body '{body_name}' not found in cache")
        return data.xmat[body_id].reshape(3, 3).copy()

    def get_torso_xpos(self, data: mujoco.MjData, agent: str) -> np.ndarray:
        """Get torso position for an agent.

        Args:
            data: MuJoCo data instance
            agent: Agent ID ('h0' or 'h1')

        Returns:
            3D position array
        """
        body_id = self.torso_body_ids.get(agent)
        if body_id is None:
            raise ValueError(f"Torso body for agent '{agent}' not found")
        return data.xpos[body_id].copy()

    def get_torso_xmat(self, data: mujoco.MjData, agent: str) -> np.ndarray:
        """Get torso rotation matrix for an agent.

        Args:
            data: MuJoCo data instance
            agent: Agent ID ('h0' or 'h1')

        Returns:
            3x3 rotation matrix
        """
        body_id = self.torso_body_ids.get(agent)
        if body_id is None:
            raise ValueError(f"Torso body for agent '{agent}' not found")
        return data.xmat[body_id].reshape(3, 3).copy()

    def get_num_actuators(self, agent: str) -> int:
        """Get number of actuators for an agent.

        Args:
            agent: Agent ID ('h0' or 'h1')

        Returns:
            Number of actuators
        """
        return len(self.actuator_idx[agent])
