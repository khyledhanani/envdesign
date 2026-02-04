"""PettingZoo Parallel environment for two-agent humanoid hugging."""

from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import functools

import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv

from humanoid_hug.utils.ids import IDCache
from humanoid_hug.utils.contacts import ContactDetector
from humanoid_hug.utils.obs import ObservationBuilder
from humanoid_hug.utils.rewards import RewardComputer


class HumanoidHugEnv(ParallelEnv):
    """Two-agent humanoid hugging environment.

    A PettingZoo ParallelEnv where two humanoid agents learn to approach,
    align, wrap arms around each other's torso, and hold a stable hug.

    Attributes:
        metadata: Environment metadata
        agents: List of agent IDs
        possible_agents: List of possible agent IDs
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "humanoid_hug_v0",
        "is_parallelizable": True,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        horizon: int = 1000,
        frame_skip: int = 5,
        hug_hold_target: int = 30,
        stage: int = 0,
    ):
        """Initialize the humanoid hug environment.

        Args:
            render_mode: Rendering mode ('human' or 'rgb_array')
            horizon: Maximum episode length in steps
            frame_skip: Number of physics steps per action
            hug_hold_target: Number of consecutive hug-condition steps for success
            stage: Curriculum stage (0-3)
        """
        super().__init__()

        self.render_mode = render_mode
        self.horizon = horizon
        self.frame_skip = frame_skip
        self.hug_hold_target = hug_hold_target
        self.stage = stage

        # Load MuJoCo model
        model_path = Path(__file__).parent / "mjcf" / "humanoid_hug.xml"
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        # Store initial state for reset
        self._initial_qpos = self.data.qpos.copy()
        self._initial_qvel = self.data.qvel.copy()

        # Initialize utilities
        self.id_cache = IDCache(self.model)
        self.contact_detector = ContactDetector(self.id_cache)
        self.obs_builder = ObservationBuilder(self.id_cache, self.model)
        self.reward_computer = RewardComputer(self.id_cache, self.model, stage)

        # Agent setup
        self.agents = ["h0", "h1"]
        self.possible_agents = self.agents.copy()

        # Determine action/observation dimensions
        self._action_dim = self.id_cache.get_num_actuators("h0")
        self._obs_dim = self.obs_builder.get_obs_dim()

        # Episode state
        self._step_count = 0
        self._hug_hold_steps = 0
        self._terminated = False
        self._truncated = False

        # Rendering
        self._viewer = None
        self._render_context = None

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:
        """Get the observation space for an agent.

        Args:
            agent: Agent ID

        Returns:
            Gymnasium observation space
        """
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:
        """Get the action space for an agent.

        Args:
            agent: Agent ID

        Returns:
            Gymnasium action space
        """
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._action_dim,),
            dtype=np.float32,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        """Reset the environment.

        Args:
            seed: Random seed
            options: Reset options. Supports:
                - 'stage': Curriculum stage (0-3)

        Returns:
            Tuple of (observations dict, infos dict)
        """
        if seed is not None:
            np.random.seed(seed)

        # Handle options
        if options is not None:
            if "stage" in options:
                self.stage = options["stage"]
                self.reward_computer.set_stage(self.stage)

        # Reset MuJoCo state
        mujoco.mj_resetData(self.model, self.data)

        # Randomize initial positions slightly
        self._randomize_initial_state()

        # Forward to compute positions
        mujoco.mj_forward(self.model, self.data)

        # Reset episode state
        self._step_count = 0
        self._hug_hold_steps = 0
        self._terminated = False
        self._truncated = False
        self.agents = self.possible_agents.copy()

        # Build observations
        contact_info = self.contact_detector.detect_contacts(self.data)
        observations = self.obs_builder.build_observations(self.data, contact_info)

        # Build infos
        infos = {
            agent: {
                "stage": self.stage,
                "weights": self.reward_computer.get_weights_dict(),
            }
            for agent in self.agents
        }

        return observations, infos

    def _randomize_initial_state(self) -> None:
        """Randomize the initial positions of the humanoids."""
        # Get qpos indices for root joints
        h0_qpos = self.id_cache.joint_qpos_idx["h0"]
        h1_qpos = self.id_cache.joint_qpos_idx["h1"]

        # Base positions: h0 at (-1, 0, 1.4), h1 at (1, 0, 1.4)
        # Randomize x position (distance apart: 1.5-2.5m)
        half_dist = np.random.uniform(0.75, 1.25)

        # H0 position
        self.data.qpos[h0_qpos[0]] = -half_dist + np.random.uniform(-0.1, 0.1)
        self.data.qpos[h0_qpos[1]] = np.random.uniform(-0.1, 0.1)
        self.data.qpos[h0_qpos[2]] = 1.4

        # H0 orientation (facing roughly toward h1, with small noise)
        # Quaternion for facing +x direction with small yaw noise
        yaw_noise = np.random.uniform(-0.2, 0.2)
        self.data.qpos[h0_qpos[3]] = np.cos(yaw_noise / 2)  # w
        self.data.qpos[h0_qpos[4]] = 0.0  # x
        self.data.qpos[h0_qpos[5]] = 0.0  # y
        self.data.qpos[h0_qpos[6]] = np.sin(yaw_noise / 2)  # z

        # H1 position
        self.data.qpos[h1_qpos[0]] = half_dist + np.random.uniform(-0.1, 0.1)
        self.data.qpos[h1_qpos[1]] = np.random.uniform(-0.1, 0.1)
        self.data.qpos[h1_qpos[2]] = 1.4

        # H1 orientation (facing roughly toward h0, with small noise)
        # Quaternion for facing -x direction
        yaw_h1 = np.pi + np.random.uniform(-0.2, 0.2)
        self.data.qpos[h1_qpos[3]] = np.cos(yaw_h1 / 2)  # w
        self.data.qpos[h1_qpos[4]] = 0.0  # x
        self.data.qpos[h1_qpos[5]] = 0.0  # y
        self.data.qpos[h1_qpos[6]] = np.sin(yaw_h1 / 2)  # z

        # Zero velocities
        self.data.qvel[:] = 0.0

    def step(
        self,
        actions: Dict[str, np.ndarray],
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        """Execute one environment step.

        Args:
            actions: Dictionary mapping agent ID to action array

        Returns:
            Tuple of (observations, rewards, terminations, truncations, infos)
        """
        if self._terminated or self._truncated:
            # Environment already done
            return self._get_terminal_returns()

        # Pack actions into control vector
        ctrl = np.zeros(self.model.nu)
        for agent in self.agents:
            if agent in actions:
                action = np.clip(actions[agent], -1.0, 1.0)
                idx = self.id_cache.actuator_idx[agent]
                ctrl[idx] = action

        self.data.ctrl[:] = ctrl

        # Step physics
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        # Check for NaN (numerical blowup)
        if np.any(np.isnan(self.data.qpos)) or np.any(np.isnan(self.data.qvel)):
            return self._handle_nan_error()

        # Detect contacts
        contact_info = self.contact_detector.detect_contacts(self.data)
        contact_force_proxy = self.contact_detector.get_contact_force_proxy(self.data)

        # Check hug condition
        hug_condition, hug_info = self.reward_computer.check_hug_condition(
            self.data, contact_info
        )

        if hug_condition:
            self._hug_hold_steps += 1
        else:
            self._hug_hold_steps = 0

        # Check for success
        success = self._hug_hold_steps >= self.hug_hold_target

        # Check for fall
        fallen, fallen_agent = self.reward_computer.check_fallen(self.data)

        # Compute reward
        reward, reward_info = self.reward_computer.compute_reward(
            self.data,
            contact_info,
            ctrl,
            contact_force_proxy,
            self._hug_hold_steps,
            success,
            fallen,
        )

        # Determine termination/truncation
        termination_reason = None
        if success:
            self._terminated = True
            termination_reason = "success"
        elif fallen:
            self._terminated = True
            termination_reason = f"fall_{fallen_agent}"
        elif self._step_count >= self.horizon:
            self._truncated = True
            termination_reason = "horizon"

        # Build observations
        observations = self.obs_builder.build_observations(self.data, contact_info)

        # Build rewards dict (same reward for both agents)
        rewards = {agent: reward for agent in self.agents}

        # Build terminations/truncations
        terminations = {agent: self._terminated for agent in self.agents}
        truncations = {agent: self._truncated for agent in self.agents}

        # Build infos
        infos = {}
        for agent in self.agents:
            infos[agent] = {
                "step": self._step_count,
                "hug_hold_steps": self._hug_hold_steps,
                "termination_reason": termination_reason,
                "stage": self.stage,
                **reward_info,
                **hug_info,
                **contact_info,
            }

        # Clear agents if terminated
        if self._terminated or self._truncated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _get_terminal_returns(self):
        """Return empty dicts when environment is already done."""
        return (
            {agent: np.zeros(self._obs_dim, dtype=np.float32) for agent in self.possible_agents},
            {agent: 0.0 for agent in self.possible_agents},
            {agent: True for agent in self.possible_agents},
            {agent: self._truncated for agent in self.possible_agents},
            {agent: {} for agent in self.possible_agents},
        )

    def _handle_nan_error(self):
        """Handle NaN in simulation state."""
        self._terminated = True
        self.agents = []
        return (
            {agent: np.zeros(self._obs_dim, dtype=np.float32) for agent in self.possible_agents},
            {agent: -100.0 for agent in self.possible_agents},
            {agent: True for agent in self.possible_agents},
            {agent: False for agent in self.possible_agents},
            {agent: {"termination_reason": "nan_error"} for agent in self.possible_agents},
        )

    def render(self) -> Optional[np.ndarray]:
        """Render the environment.

        Returns:
            RGB array if render_mode is 'rgb_array', else None
        """
        if self.render_mode is None:
            return None

        if self.render_mode == "human":
            return self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()

        return None

    def _render_human(self) -> None:
        """Render to a human-viewable window."""
        if self._viewer is None:
            # On macOS, launch_passive requires mjpython.
            # Try launch_passive first, then fall back to blocking launch.
            try:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self._viewer_is_passive = True
            except Exception:
                # Fallback: use blocking launch (will block until window closes)
                # This won't work well in a loop, so we use a simple renderer approach
                self._viewer_is_passive = False
                try:
                    # Create an offscreen renderer for display if passive fails
                    if self._render_context is None:
                        self._render_context = mujoco.Renderer(self.model, 480, 640)
                    import warnings
                    if not hasattr(self, '_viewer_warning_shown'):
                        warnings.warn(
                            "Passive viewer unavailable. On macOS, run with 'mjpython' "
                            "instead of 'python3' for interactive rendering, "
                            "or use --mode video to save a video file."
                        )
                        self._viewer_warning_shown = True
                except Exception:
                    pass

        if self._viewer is not None and self._viewer_is_passive:
            self._viewer.sync()

    def _render_rgb_array(self) -> np.ndarray:
        """Render to an RGB array.

        Returns:
            RGB image array of shape (height, width, 3)
        """
        width, height = 640, 480

        if self._render_context is None:
            self._render_context = mujoco.Renderer(self.model, height, width)

        self._render_context.update_scene(self.data, camera="h0_track")
        return self._render_context.render()

    def close(self) -> None:
        """Clean up resources."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

        if self._render_context is not None:
            self._render_context.close()
            self._render_context = None

    def state(self) -> np.ndarray:
        """Get the global state (for centralized critic if needed).

        Returns:
            Full simulation state
        """
        return np.concatenate([
            self.data.qpos.copy(),
            self.data.qvel.copy(),
        ])

    @property
    def max_num_agents(self) -> int:
        """Maximum number of agents."""
        return 2

    @property
    def num_agents(self) -> int:
        """Current number of active agents."""
        return len(self.agents)
