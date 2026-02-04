"""Reward computation for the humanoid hug environment."""

from typing import Dict, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import mujoco

from humanoid_hug.utils.ids import IDCache
from humanoid_hug.utils.kinematics import (
    get_forward_vector,
    get_up_vector,
    compute_facing_alignment,
    compute_tilt_angle,
    get_root_linear_velocity,
)


@dataclass
class CurriculumWeights:
    """Reward weights for different curriculum stages."""

    # Shaping rewards
    w_distance: float = 1.0
    w_facing: float = 0.5
    w_stability: float = 0.3

    # Embrace rewards
    w_contact: float = 0.0
    w_hand_to_back: float = 0.0

    # Penalties
    w_fall: float = -100.0
    w_impact: float = -0.01
    w_energy: float = -0.001

    # Success bonus
    r_success: float = 100.0


# Default curriculum stages
CURRICULUM_STAGES: Dict[int, CurriculumWeights] = {
    0: CurriculumWeights(
        w_distance=2.0, w_facing=1.0, w_stability=0.5,
        w_contact=0.0, w_hand_to_back=0.0,
        w_fall=-100.0, w_impact=-0.01, w_energy=-0.001,
        r_success=100.0
    ),
    1: CurriculumWeights(
        w_distance=1.5, w_facing=1.0, w_stability=0.5,
        w_contact=0.0, w_hand_to_back=2.0,
        w_fall=-100.0, w_impact=-0.01, w_energy=-0.001,
        r_success=100.0
    ),
    2: CurriculumWeights(
        w_distance=1.0, w_facing=0.8, w_stability=0.5,
        w_contact=3.0, w_hand_to_back=1.0,
        w_fall=-100.0, w_impact=-0.01, w_energy=-0.001,
        r_success=150.0
    ),
    3: CurriculumWeights(
        w_distance=0.5, w_facing=0.5, w_stability=0.3,
        w_contact=5.0, w_hand_to_back=0.5,
        w_fall=-100.0, w_impact=-0.01, w_energy=-0.001,
        r_success=200.0
    ),
}


class RewardComputer:
    """Computes rewards for the humanoid hug environment."""

    # Hug condition thresholds
    D_MIN = 0.25  # Minimum chest distance for hug
    D_MAX = 0.60  # Maximum chest distance for hug
    FACING_THRESH = 0.6  # Minimum facing alignment
    V_THRESH = 1.0  # Maximum relative velocity for stable hug
    TILT_THRESH = 0.5  # Maximum tilt from vertical (radians, ~30 degrees)
    FALL_HEIGHT_THRESH = 0.5  # Minimum torso height before considered fallen

    # Reward shaping parameters
    ALPHA_DIST = 3.0  # Distance reward decay
    BETA_SPEED = 2.0  # Stability reward decay

    def __init__(self, id_cache: IDCache, model: mujoco.MjModel, stage: int = 0):
        """Initialize the reward computer.

        Args:
            id_cache: Cached IDs from the MuJoCo model
            model: MuJoCo model instance
            stage: Curriculum stage (0-3)
        """
        self.id_cache = id_cache
        self.model = model
        self.set_stage(stage)

    def set_stage(self, stage: int) -> None:
        """Set the curriculum stage.

        Args:
            stage: Curriculum stage (0-3)
        """
        self.stage = stage
        self.weights = CURRICULUM_STAGES.get(stage, CURRICULUM_STAGES[0])

    def get_weights_dict(self) -> Dict[str, float]:
        """Get current weights as a dictionary for logging.

        Returns:
            Dictionary of weight names to values
        """
        return {
            "w_distance": self.weights.w_distance,
            "w_facing": self.weights.w_facing,
            "w_stability": self.weights.w_stability,
            "w_contact": self.weights.w_contact,
            "w_hand_to_back": self.weights.w_hand_to_back,
            "w_fall": self.weights.w_fall,
            "w_impact": self.weights.w_impact,
            "w_energy": self.weights.w_energy,
            "r_success": self.weights.r_success,
        }

    def compute_reward(
        self,
        data: mujoco.MjData,
        contact_info: Dict[str, bool],
        ctrl: np.ndarray,
        contact_force_proxy: float,
        hug_hold_steps: int,
        success: bool,
        fallen: bool,
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute the reward for the current timestep.

        Both agents receive the same reward (cooperative task).

        Args:
            data: MuJoCo data instance
            contact_info: Contact detection results
            ctrl: Control vector
            contact_force_proxy: Proxy for contact forces
            hug_hold_steps: Number of consecutive hug-condition steps
            success: Whether hug success was achieved this step
            fallen: Whether either agent has fallen

        Returns:
            Tuple of (reward, info_dict with reward components)
        """
        info = {}
        total_reward = 0.0

        # Get torso positions and orientations
        h0_xpos = self.id_cache.get_torso_xpos(data, "h0")
        h1_xpos = self.id_cache.get_torso_xpos(data, "h1")
        h0_xmat = self.id_cache.get_torso_xmat(data, "h0")
        h1_xmat = self.id_cache.get_torso_xmat(data, "h1")

        # Distance between chests
        try:
            h0_chest = self.id_cache.get_site_xpos(data, "h0_chest")
            h1_chest = self.id_cache.get_site_xpos(data, "h1_chest")
        except ValueError:
            h0_chest = h0_xpos
            h1_chest = h1_xpos

        distance = np.linalg.norm(h0_chest - h1_chest)
        info["chest_distance"] = distance

        # === Distance reward ===
        r_dist = np.exp(-self.ALPHA_DIST * distance)
        r_dist_weighted = self.weights.w_distance * r_dist
        total_reward += r_dist_weighted
        info["r_distance"] = r_dist_weighted

        # === Facing reward ===
        h0_fwd = get_forward_vector(h0_xmat)
        h1_fwd = get_forward_vector(h1_xmat)
        facing = compute_facing_alignment(h0_fwd, h1_fwd)
        r_facing = max(0.0, facing)
        r_facing_weighted = self.weights.w_facing * r_facing
        total_reward += r_facing_weighted
        info["facing_alignment"] = facing
        info["r_facing"] = r_facing_weighted

        # === Stability reward ===
        h0_qvel_idx = self.id_cache.joint_qvel_idx["h0"]
        h1_qvel_idx = self.id_cache.joint_qvel_idx["h1"]
        h0_vel = get_root_linear_velocity(data, h0_qvel_idx)
        h1_vel = get_root_linear_velocity(data, h1_qvel_idx)
        rel_speed = np.linalg.norm(h0_vel - h1_vel)
        r_stability = np.exp(-self.BETA_SPEED * rel_speed)
        r_stability_weighted = self.weights.w_stability * r_stability
        total_reward += r_stability_weighted
        info["relative_speed"] = rel_speed
        info["r_stability"] = r_stability_weighted

        # === Contact reward ===
        n_contacts = int(contact_info.get("h0_arm_torso", False)) + \
                     int(contact_info.get("h1_arm_torso", False))
        r_contact_weighted = self.weights.w_contact * n_contacts
        total_reward += r_contact_weighted
        info["r_contact"] = r_contact_weighted
        info["n_arm_torso_contacts"] = n_contacts

        # === Hand-to-back shaping ===
        if self.weights.w_hand_to_back > 0:
            r_hand_back = self._compute_hand_to_back_reward(data)
            r_hand_back_weighted = self.weights.w_hand_to_back * r_hand_back
            total_reward += r_hand_back_weighted
            info["r_hand_to_back"] = r_hand_back_weighted
        else:
            info["r_hand_to_back"] = 0.0

        # === Energy penalty ===
        energy = np.sum(np.square(ctrl))
        r_energy = self.weights.w_energy * energy
        total_reward += r_energy
        info["r_energy"] = r_energy
        info["energy"] = energy

        # === Impact penalty ===
        r_impact = self.weights.w_impact * contact_force_proxy
        total_reward += r_impact
        info["r_impact"] = r_impact

        # === Fall penalty ===
        if fallen:
            r_fall = self.weights.w_fall
            total_reward += r_fall
            info["r_fall"] = r_fall
        else:
            info["r_fall"] = 0.0

        # === Success bonus ===
        if success:
            total_reward += self.weights.r_success
            info["r_success"] = self.weights.r_success
        else:
            info["r_success"] = 0.0

        info["total_reward"] = total_reward
        info["hug_hold_steps"] = hug_hold_steps

        return total_reward, info

    def _compute_hand_to_back_reward(self, data: mujoco.MjData) -> float:
        """Compute reward for hands being near partner's back.

        Args:
            data: MuJoCo data instance

        Returns:
            Reward value (higher = hands closer to partner backs)
        """
        total = 0.0
        alpha = 4.0  # Decay rate

        try:
            h0_lhand = self.id_cache.get_site_xpos(data, "h0_lhand")
            h0_rhand = self.id_cache.get_site_xpos(data, "h0_rhand")
            h1_lhand = self.id_cache.get_site_xpos(data, "h1_lhand")
            h1_rhand = self.id_cache.get_site_xpos(data, "h1_rhand")
            h0_back = self.id_cache.get_site_xpos(data, "h0_back")
            h1_back = self.id_cache.get_site_xpos(data, "h1_back")

            # h0 hands to h1 back
            d_h0l_h1b = np.linalg.norm(h0_lhand - h1_back)
            d_h0r_h1b = np.linalg.norm(h0_rhand - h1_back)

            # h1 hands to h0 back
            d_h1l_h0b = np.linalg.norm(h1_lhand - h0_back)
            d_h1r_h0b = np.linalg.norm(h1_rhand - h0_back)

            total = (
                np.exp(-alpha * d_h0l_h1b) +
                np.exp(-alpha * d_h0r_h1b) +
                np.exp(-alpha * d_h1l_h0b) +
                np.exp(-alpha * d_h1r_h0b)
            )
        except ValueError:
            pass

        return total

    def check_hug_condition(
        self,
        data: mujoco.MjData,
        contact_info: Dict[str, bool]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if the hug condition is satisfied.

        Args:
            data: MuJoCo data instance
            contact_info: Contact detection results

        Returns:
            Tuple of (hug_condition_met, debug_info)
        """
        info = {}

        # Get torso positions
        h0_xpos = self.id_cache.get_torso_xpos(data, "h0")
        h1_xpos = self.id_cache.get_torso_xpos(data, "h1")
        h0_xmat = self.id_cache.get_torso_xmat(data, "h0")
        h1_xmat = self.id_cache.get_torso_xmat(data, "h1")

        # Check chest distance
        try:
            h0_chest = self.id_cache.get_site_xpos(data, "h0_chest")
            h1_chest = self.id_cache.get_site_xpos(data, "h1_chest")
        except ValueError:
            h0_chest = h0_xpos
            h1_chest = h1_xpos

        distance = np.linalg.norm(h0_chest - h1_chest)
        dist_ok = self.D_MIN < distance < self.D_MAX
        info["hug_dist"] = distance
        info["hug_dist_ok"] = dist_ok

        # Check facing alignment
        h0_fwd = get_forward_vector(h0_xmat)
        h1_fwd = get_forward_vector(h1_xmat)
        facing = compute_facing_alignment(h0_fwd, h1_fwd)
        facing_ok = facing > self.FACING_THRESH
        info["hug_facing"] = facing
        info["hug_facing_ok"] = facing_ok

        # Check mutual arm-torso contact
        h0_arm_h1_torso = contact_info.get("h0_arm_torso", False)
        h1_arm_h0_torso = contact_info.get("h1_arm_torso", False)
        contact_ok = h0_arm_h1_torso and h1_arm_h0_torso
        info["hug_contact_ok"] = contact_ok

        # Check relative speed
        h0_qvel_idx = self.id_cache.joint_qvel_idx["h0"]
        h1_qvel_idx = self.id_cache.joint_qvel_idx["h1"]
        h0_vel = get_root_linear_velocity(data, h0_qvel_idx)
        h1_vel = get_root_linear_velocity(data, h1_qvel_idx)
        rel_speed = np.linalg.norm(h0_vel - h1_vel)
        speed_ok = rel_speed < self.V_THRESH
        info["hug_rel_speed"] = rel_speed
        info["hug_speed_ok"] = speed_ok

        # Check uprightness
        h0_up = get_up_vector(h0_xmat)
        h1_up = get_up_vector(h1_xmat)
        h0_tilt = compute_tilt_angle(h0_up)
        h1_tilt = compute_tilt_angle(h1_up)
        upright_ok = h0_tilt < self.TILT_THRESH and h1_tilt < self.TILT_THRESH
        info["h0_tilt"] = h0_tilt
        info["h1_tilt"] = h1_tilt
        info["hug_upright_ok"] = upright_ok

        # Overall hug condition
        hug_condition = dist_ok and facing_ok and contact_ok and speed_ok and upright_ok
        info["hug_condition"] = hug_condition

        return hug_condition, info

    def check_fallen(self, data: mujoco.MjData) -> Tuple[bool, str]:
        """Check if either humanoid has fallen.

        Args:
            data: MuJoCo data instance

        Returns:
            Tuple of (fallen, which_agent_or_empty)
        """
        for agent in ["h0", "h1"]:
            xpos = self.id_cache.get_torso_xpos(data, agent)
            xmat = self.id_cache.get_torso_xmat(data, agent)
            up_vec = get_up_vector(xmat)
            tilt = compute_tilt_angle(up_vec)

            # Check height
            if xpos[2] < self.FALL_HEIGHT_THRESH:
                return True, agent

            # Check tilt (fallen if tilted too much)
            if tilt > np.pi / 2:  # More than 90 degrees
                return True, agent

        return False, ""
