#!/usr/bin/env python3
"""
MJX-accelerated IPPO training for Humanoid Hug environment.

Runs thousands of environments in parallel on GPU using MuJoCo XLA (MJX).
Achieves 50-100x speedup over CPU-based training.

Requirements:
    pip install "jax[cuda12]" flax optax mujoco>=3.0.0

Usage:
    python scripts/train_mjx.py --num-envs 4096 --total-timesteps 20_000_000
"""

import argparse
import functools
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
import mujoco
from mujoco import mjx
import numpy as np
import optax
from flax.training.train_state import TrainState
from flax import struct

# Check JAX backend
print(f"JAX devices: {jax.devices()}")
print(f"JAX default backend: {jax.default_backend()}")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MJXConfig:
    """Configuration for MJX PPO training."""
    # Environment
    horizon: int = 1000
    frame_skip: int = 5
    stage: int = 0
    
    # Training
    total_timesteps: int = 20_000_000
    num_envs: int = 4096  # Batch size on GPU
    num_steps: int = 64   # Shorter rollouts work better with many envs
    num_epochs: int = 4   # PPO epochs per update
    minibatch_size: int = 4096
    
    # PPO
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Network
    hidden_sizes: Tuple[int, ...] = (256, 256)
    
    # Logging
    log_interval: int = 10
    save_interval: int = 100
    
    # Misc
    seed: int = 42
    
    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps
    
    @property
    def num_minibatches(self) -> int:
        return max(1, self.batch_size // self.minibatch_size)
    
    @property
    def num_updates(self) -> int:
        return self.total_timesteps // self.batch_size


# =============================================================================
# MJX Environment
# =============================================================================

class EnvState(NamedTuple):
    """State for the batched MJX environment."""
    mjx_data: mjx.Data
    step_count: jax.Array
    hug_hold_steps: jax.Array
    rng: jax.Array


class HumanoidHugMJX:
    """
    MJX-accelerated two-humanoid hugging environment.
    
    Runs batched physics simulation entirely on GPU.
    """
    
    def __init__(
        self,
        model_path: str,
        num_envs: int,
        horizon: int = 1000,
        frame_skip: int = 5,
        stage: int = 0,
    ):
        self.num_envs = num_envs
        self.horizon = horizon
        self.frame_skip = frame_skip
        self.stage = stage
        
        # Load MuJoCo model
        self.mj_model = mujoco.MjModel.from_xml_path(model_path)
        self.mjx_model = mjx.put_model(self.mj_model)
        
        # Cache important indices
        self._cache_indices()
        
        # Observation and action dimensions
        self.obs_dim = self._compute_obs_dim()
        self.act_dim = len(self.h0_actuator_idx)
        
        # Reward weights based on stage
        self._set_reward_weights()
        
        print(f"  MJX Env: obs_dim={self.obs_dim}, act_dim={self.act_dim}")
        print(f"  Stage {stage} reward weights loaded")
    
    def _cache_indices(self):
        """Cache joint and actuator indices for both humanoids."""
        model = self.mj_model
        
        # Find joint indices (qpos)
        # Freejoint: 7 values (3 pos + 4 quat)
        # Hinge: 1 value
        h0_joints = []
        h1_joints = []
        for i in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name and name.startswith("h0_"):
                h0_joints.append(i)
            elif name and name.startswith("h1_"):
                h1_joints.append(i)
        
        # Get qpos/qvel indices
        self.h0_qpos_start = model.jnt_qposadr[h0_joints[0]]
        self.h1_qpos_start = model.jnt_qposadr[h1_joints[0]]
        self.h0_qvel_start = model.jnt_dofadr[h0_joints[0]]
        self.h1_qvel_start = model.jnt_dofadr[h1_joints[0]]
        
        # Count DoFs per humanoid (freejoint=6 DoF + hinge joints)
        self.nq_per_humanoid = 7 + (len(h0_joints) - 1)  # qpos: 7 for freejoint + hinges
        self.nv_per_humanoid = 6 + (len(h0_joints) - 1)  # qvel: 6 for freejoint + hinges
        
        # Actuator indices
        self.h0_actuator_idx = []
        self.h1_actuator_idx = []
        for i in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name and name.startswith("h0_"):
                self.h0_actuator_idx.append(i)
            elif name and name.startswith("h1_"):
                self.h1_actuator_idx.append(i)
        
        self.h0_actuator_idx = jnp.array(self.h0_actuator_idx)
        self.h1_actuator_idx = jnp.array(self.h1_actuator_idx)
        
        # Body indices for torso
        self.h0_torso_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "h0_torso")
        self.h1_torso_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "h1_torso")
        
        # Geom indices for contact detection
        self._cache_geom_indices()
    
    def _cache_geom_indices(self):
        """Cache geom indices for arm-torso contact detection."""
        model = self.mj_model
        
        self.h0_arm_geoms = []
        self.h1_arm_geoms = []
        self.h0_torso_geoms = []
        self.h1_torso_geoms = []
        
        arm_keywords = ["upper_arm", "lower_arm", "hand"]
        torso_keywords = ["torso", "chest", "abdomen", "pelvis"]
        
        for i in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name is None:
                continue
            
            if name.startswith("h0_"):
                if any(kw in name for kw in arm_keywords):
                    self.h0_arm_geoms.append(i)
                elif any(kw in name for kw in torso_keywords):
                    self.h0_torso_geoms.append(i)
            elif name.startswith("h1_"):
                if any(kw in name for kw in arm_keywords):
                    self.h1_arm_geoms.append(i)
                elif any(kw in name for kw in torso_keywords):
                    self.h1_torso_geoms.append(i)
        
        self.h0_arm_geoms = jnp.array(self.h0_arm_geoms)
        self.h1_arm_geoms = jnp.array(self.h1_arm_geoms)
        self.h0_torso_geoms = jnp.array(self.h0_torso_geoms)
        self.h1_torso_geoms = jnp.array(self.h1_torso_geoms)
    
    def _compute_obs_dim(self) -> int:
        """Compute observation dimension per agent."""
        # Proprioception: joint pos (excl root pos) + joint vel (excl root lin) + root quat + root angvel
        proprio = (self.nq_per_humanoid - 3) + (self.nv_per_humanoid - 3) + 4 + 3
        # Partner features: rel chest pos + rel pelvis pos + rel vel + facing + up align
        partner = 3 + 3 + 3 + 1 + 1
        # Contact bits: l_arm, r_arm, partner_arm
        contact = 3
        return proprio + partner + contact
    
    def _set_reward_weights(self):
        """Set reward weights based on curriculum stage."""
        stages = {
            0: {"dist": 2.0, "facing": 1.0, "stability": 0.5, "contact": 0.0, "hand_back": 0.0, "success": 100.0},
            1: {"dist": 1.5, "facing": 1.0, "stability": 0.5, "contact": 0.0, "hand_back": 2.0, "success": 100.0},
            2: {"dist": 1.0, "facing": 0.8, "stability": 0.5, "contact": 3.0, "hand_back": 1.0, "success": 150.0},
            3: {"dist": 0.5, "facing": 0.5, "stability": 0.3, "contact": 5.0, "hand_back": 0.5, "success": 200.0},
        }
        self.weights = stages.get(self.stage, stages[0])
    
    def reset(self, rng: jax.Array) -> Tuple[EnvState, Dict[str, jax.Array]]:
        """Reset all environments."""
        rng, rng_reset = jax.random.split(rng)
        
        # Create batched reset RNGs
        batch_rngs = jax.random.split(rng_reset, self.num_envs)
        
        # Vectorized reset - vmap over RNG, model is static
        batched_reset = jax.vmap(self._reset_single, in_axes=(0,))
        mjx_data = batched_reset(batch_rngs)
        
        state = EnvState(
            mjx_data=mjx_data,
            step_count=jnp.zeros(self.num_envs, dtype=jnp.int32),
            hug_hold_steps=jnp.zeros(self.num_envs, dtype=jnp.int32),
            rng=rng,
        )
        
        obs = self._get_obs(mjx_data)
        return state, obs
    
    def _reset_single(self, rng: jax.Array) -> mjx.Data:
        """Reset a single environment with randomized initial state."""
        rng, rng_dist, rng_h0_pos, rng_h1_pos, rng_h0_yaw, rng_h1_yaw = jax.random.split(rng, 6)
        
        # Create fresh data for this env
        base_data = mjx.make_data(self.mjx_model)
        
        qpos = base_data.qpos.copy()
        qvel = jnp.zeros_like(base_data.qvel)
        
        # Randomize initial distance
        half_dist = jax.random.uniform(rng_dist, minval=0.75, maxval=1.25)
        
        # H0 position (facing +x)
        h0_x = -half_dist + jax.random.uniform(rng_h0_pos, minval=-0.1, maxval=0.1)
        h0_y = jax.random.uniform(rng_h0_pos, minval=-0.1, maxval=0.1)
        yaw_h0 = jax.random.uniform(rng_h0_yaw, minval=-0.2, maxval=0.2)
        
        qpos = qpos.at[self.h0_qpos_start].set(h0_x)
        qpos = qpos.at[self.h0_qpos_start + 1].set(h0_y)
        qpos = qpos.at[self.h0_qpos_start + 2].set(1.4)
        qpos = qpos.at[self.h0_qpos_start + 3].set(jnp.cos(yaw_h0 / 2))
        qpos = qpos.at[self.h0_qpos_start + 4].set(0.0)
        qpos = qpos.at[self.h0_qpos_start + 5].set(0.0)
        qpos = qpos.at[self.h0_qpos_start + 6].set(jnp.sin(yaw_h0 / 2))
        
        # H1 position (facing -x)
        h1_x = half_dist + jax.random.uniform(rng_h1_pos, minval=-0.1, maxval=0.1)
        h1_y = jax.random.uniform(rng_h1_pos, minval=-0.1, maxval=0.1)
        yaw_h1 = jnp.pi + jax.random.uniform(rng_h1_yaw, minval=-0.2, maxval=0.2)
        
        qpos = qpos.at[self.h1_qpos_start].set(h1_x)
        qpos = qpos.at[self.h1_qpos_start + 1].set(h1_y)
        qpos = qpos.at[self.h1_qpos_start + 2].set(1.4)
        qpos = qpos.at[self.h1_qpos_start + 3].set(jnp.cos(yaw_h1 / 2))
        qpos = qpos.at[self.h1_qpos_start + 4].set(0.0)
        qpos = qpos.at[self.h1_qpos_start + 5].set(0.0)
        qpos = qpos.at[self.h1_qpos_start + 6].set(jnp.sin(yaw_h1 / 2))
        
        new_data = base_data.replace(qpos=qpos, qvel=qvel)
        return mjx.forward(self.mjx_model, new_data)
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: EnvState,
        h0_actions: jax.Array,
        h1_actions: jax.Array,
    ) -> Tuple[EnvState, Dict[str, jax.Array], jax.Array, jax.Array, Dict[str, jax.Array]]:
        """
        Step all environments in parallel.
        
        Args:
            state: Current environment state
            h0_actions: Actions for humanoid 0, shape (num_envs, act_dim)
            h1_actions: Actions for humanoid 1, shape (num_envs, act_dim)
        
        Returns:
            new_state, obs_dict, rewards, dones, info_dict
        """
        # Combine actions into control vector
        ctrl = jnp.zeros((self.num_envs, self.mj_model.nu))
        ctrl = ctrl.at[:, self.h0_actuator_idx].set(jnp.clip(h0_actions, -1.0, 1.0))
        ctrl = ctrl.at[:, self.h1_actuator_idx].set(jnp.clip(h1_actions, -1.0, 1.0))
        
        # Update control in batched data
        mjx_data = state.mjx_data.replace(ctrl=ctrl)
        
        # Vmapped physics step function
        @jax.vmap
        def step_single(data):
            """Step a single environment through frame_skip steps."""
            def do_step(d, _):
                return mjx.step(self.mjx_model, d), None
            d, _ = jax.lax.scan(do_step, data, None, length=self.frame_skip)
            return d
        
        mjx_data = step_single(mjx_data)
        
        # Compute observations, rewards, terminations
        obs = self._get_obs(mjx_data)
        
        # Check hug condition
        hug_condition = self._check_hug_condition(mjx_data)
        new_hug_hold = jnp.where(hug_condition, state.hug_hold_steps + 1, 0)
        
        # Check terminations
        fallen = self._check_fallen(mjx_data)
        success = new_hug_hold >= 30  # hug_hold_target
        
        # Compute rewards
        rewards = self._compute_reward(mjx_data, ctrl, new_hug_hold, success, fallen)
        
        # Determine done
        new_step_count = state.step_count + 1
        truncated = new_step_count >= self.horizon
        terminated = fallen | success
        done = terminated | truncated
        
        # Auto-reset done environments
        rng, rng_reset = jax.random.split(state.rng)
        reset_rngs = jax.random.split(rng_reset, self.num_envs)
        
        def maybe_reset(done_flag, data, reset_rng):
            reset_data = self._reset_single(reset_rng)
            return jax.lax.cond(done_flag, lambda: reset_data, lambda: data)
        
        mjx_data = jax.vmap(maybe_reset)(done, mjx_data, reset_rngs)
        
        # Reset counters for done envs
        new_step_count = jnp.where(done, 0, new_step_count)
        new_hug_hold = jnp.where(done, 0, new_hug_hold)
        
        new_state = EnvState(
            mjx_data=mjx_data,
            step_count=new_step_count,
            hug_hold_steps=new_hug_hold,
            rng=rng,
        )
        
        info = {
            "success": success,
            "fallen": fallen,
            "hug_hold_steps": state.hug_hold_steps,  # Pre-reset value
        }
        
        return new_state, obs, rewards, done, info
    
    def _get_obs(self, mjx_data: mjx.Data) -> Dict[str, jax.Array]:
        """Build observations for both agents."""
        h0_obs = jax.vmap(lambda d: self._get_agent_obs(d, "h0"))(mjx_data)
        h1_obs = jax.vmap(lambda d: self._get_agent_obs(d, "h1"))(mjx_data)
        return {"h0": h0_obs, "h1": h1_obs}
    
    def _get_agent_obs(self, data: mjx.Data, agent: str) -> jax.Array:
        """Build observation for a single agent in a single env."""
        partner = "h1" if agent == "h0" else "h0"
        
        if agent == "h0":
            qpos_start = self.h0_qpos_start
            qvel_start = self.h0_qvel_start
            torso_idx = self.h0_torso_idx
            partner_qpos_start = self.h1_qpos_start
            partner_qvel_start = self.h1_qvel_start
            partner_torso_idx = self.h1_torso_idx
        else:
            qpos_start = self.h1_qpos_start
            qvel_start = self.h1_qvel_start
            torso_idx = self.h1_torso_idx
            partner_qpos_start = self.h0_qpos_start
            partner_qvel_start = self.h0_qvel_start
            partner_torso_idx = self.h0_torso_idx
        
        obs_parts = []
        
        # Joint positions (excluding root position, keeping root quat + joint angles)
        joint_qpos = data.qpos[qpos_start + 3: qpos_start + self.nq_per_humanoid]
        obs_parts.append(joint_qpos)
        
        # Joint velocities (excluding root linear velocity)
        joint_qvel = data.qvel[qvel_start + 3: qvel_start + self.nv_per_humanoid]
        obs_parts.append(joint_qvel)
        
        # Root quaternion
        root_quat = data.qpos[qpos_start + 3: qpos_start + 7]
        obs_parts.append(root_quat)
        
        # Root angular velocity
        root_angvel = data.qvel[qvel_start + 3: qvel_start + 6]
        obs_parts.append(root_angvel)
        
        # Partner relative features
        self_pos = data.xpos[torso_idx]
        self_mat = data.xmat[torso_idx].reshape(3, 3)
        partner_pos = data.xpos[partner_torso_idx]
        
        # Relative position (chest approximation)
        rel_pos = partner_pos - self_pos
        rel_pos_local = self_mat.T @ rel_pos
        obs_parts.append(rel_pos_local)
        
        # Relative pelvis position (approximation: partner_pos - [0,0,0.2])
        partner_pelvis = partner_pos - jnp.array([0.0, 0.0, 0.2])
        rel_pelvis = partner_pelvis - self_pos
        rel_pelvis_local = self_mat.T @ rel_pelvis
        obs_parts.append(rel_pelvis_local)
        
        # Relative velocity
        self_vel = data.qvel[qvel_start: qvel_start + 3]
        partner_vel = data.qvel[partner_qvel_start: partner_qvel_start + 3]
        rel_vel = partner_vel - self_vel
        rel_vel_local = self_mat.T @ rel_vel
        obs_parts.append(rel_vel_local)
        
        # Facing alignment
        self_fwd = self_mat[:, 0]  # x-axis of rotation matrix
        partner_mat = data.xmat[partner_torso_idx].reshape(3, 3)
        partner_fwd = partner_mat[:, 0]
        facing = -jnp.dot(self_fwd, partner_fwd)  # Negative because facing each other
        obs_parts.append(jnp.array([facing]))
        
        # Up alignment
        self_up = self_mat[:, 2]
        partner_up = partner_mat[:, 2]
        up_align = jnp.dot(self_up, partner_up)
        obs_parts.append(jnp.array([up_align]))
        
        # Contact bits (simplified - using distance proxy)
        # Real contact detection in MJX is complex, use distance as proxy
        distance = jnp.linalg.norm(rel_pos)
        arm_contact_proxy = (distance < 0.6).astype(jnp.float32)
        obs_parts.append(jnp.array([arm_contact_proxy, arm_contact_proxy, arm_contact_proxy]))
        
        return jnp.concatenate(obs_parts)
    
    def _check_hug_condition(self, mjx_data: mjx.Data) -> jax.Array:
        """Check if hug condition is met for each env."""
        def check_single(data):
            h0_pos = data.xpos[self.h0_torso_idx]
            h1_pos = data.xpos[self.h1_torso_idx]
            h0_mat = data.xmat[self.h0_torso_idx].reshape(3, 3)
            h1_mat = data.xmat[self.h1_torso_idx].reshape(3, 3)
            
            # Distance
            distance = jnp.linalg.norm(h0_pos - h1_pos)
            dist_ok = (distance > 0.25) & (distance < 0.60)
            
            # Facing
            h0_fwd = h0_mat[:, 0]
            h1_fwd = h1_mat[:, 0]
            facing = -jnp.dot(h0_fwd, h1_fwd)
            facing_ok = facing > 0.6
            
            # Stability
            h0_vel = data.qvel[self.h0_qvel_start: self.h0_qvel_start + 3]
            h1_vel = data.qvel[self.h1_qvel_start: self.h1_qvel_start + 3]
            rel_speed = jnp.linalg.norm(h0_vel - h1_vel)
            speed_ok = rel_speed < 1.0
            
            # Upright
            h0_up = h0_mat[:, 2]
            h1_up = h1_mat[:, 2]
            h0_tilt = jnp.arccos(jnp.clip(h0_up[2], -1, 1))
            h1_tilt = jnp.arccos(jnp.clip(h1_up[2], -1, 1))
            upright_ok = (h0_tilt < 0.5) & (h1_tilt < 0.5)
            
            # Contact (distance proxy for GPU efficiency)
            contact_ok = distance < 0.55
            
            return dist_ok & facing_ok & speed_ok & upright_ok & contact_ok
        
        return jax.vmap(check_single)(mjx_data)
    
    def _check_fallen(self, mjx_data: mjx.Data) -> jax.Array:
        """Check if either humanoid has fallen."""
        def check_single(data):
            h0_z = data.xpos[self.h0_torso_idx, 2]
            h1_z = data.xpos[self.h1_torso_idx, 2]
            
            h0_mat = data.xmat[self.h0_torso_idx].reshape(3, 3)
            h1_mat = data.xmat[self.h1_torso_idx].reshape(3, 3)
            h0_tilt = jnp.arccos(jnp.clip(h0_mat[2, 2], -1, 1))
            h1_tilt = jnp.arccos(jnp.clip(h1_mat[2, 2], -1, 1))
            
            height_fall = (h0_z < 0.5) | (h1_z < 0.5)
            tilt_fall = (h0_tilt > jnp.pi / 2) | (h1_tilt > jnp.pi / 2)
            
            return height_fall | tilt_fall
        
        return jax.vmap(check_single)(mjx_data)
    
    def _compute_reward(
        self,
        mjx_data: mjx.Data,
        ctrl: jax.Array,
        hug_hold_steps: jax.Array,
        success: jax.Array,
        fallen: jax.Array,
    ) -> jax.Array:
        """Compute rewards for all environments."""
        def compute_single(data, ctrl_single, hug_hold, succ, fall):
            h0_pos = data.xpos[self.h0_torso_idx]
            h1_pos = data.xpos[self.h1_torso_idx]
            h0_mat = data.xmat[self.h0_torso_idx].reshape(3, 3)
            h1_mat = data.xmat[self.h1_torso_idx].reshape(3, 3)
            
            # Distance reward
            distance = jnp.linalg.norm(h0_pos - h1_pos)
            r_dist = self.weights["dist"] * jnp.exp(-3.0 * distance)
            
            # Facing reward
            h0_fwd = h0_mat[:, 0]
            h1_fwd = h1_mat[:, 0]
            facing = -jnp.dot(h0_fwd, h1_fwd)
            r_facing = self.weights["facing"] * jnp.maximum(0.0, facing)
            
            # Stability reward
            h0_vel = data.qvel[self.h0_qvel_start: self.h0_qvel_start + 3]
            h1_vel = data.qvel[self.h1_qvel_start: self.h1_qvel_start + 3]
            rel_speed = jnp.linalg.norm(h0_vel - h1_vel)
            r_stability = self.weights["stability"] * jnp.exp(-2.0 * rel_speed)
            
            # Contact reward (distance proxy)
            n_contacts = (distance < 0.55).astype(jnp.float32) * 2
            r_contact = self.weights["contact"] * n_contacts
            
            # Energy penalty
            energy = jnp.sum(jnp.square(ctrl_single))
            r_energy = -0.001 * energy
            
            # Fall penalty
            r_fall = jnp.where(fall, -100.0, 0.0)
            
            # Success bonus
            r_success = jnp.where(succ, self.weights["success"], 0.0)
            
            total = r_dist + r_facing + r_stability + r_contact + r_energy + r_fall + r_success
            return total
        
        return jax.vmap(compute_single)(mjx_data, ctrl, hug_hold_steps, success, fallen)


# =============================================================================
# Neural Network (Flax)
# =============================================================================

class PPONetwork(nn.Module):
    """Actor-Critic network for PPO using Flax."""
    hidden_sizes: Tuple[int, ...]
    act_dim: int
    
    @nn.compact
    def __call__(self, x):
        # Actor
        actor = x
        for size in self.hidden_sizes:
            actor = nn.Dense(size)(actor)
            actor = nn.tanh(actor)
        action_mean = nn.Dense(self.act_dim)(actor)
        action_logstd = self.param("logstd", nn.initializers.zeros, (self.act_dim,))
        
        # Critic
        critic = x
        for size in self.hidden_sizes:
            critic = nn.Dense(size)(critic)
            critic = nn.tanh(critic)
        value = nn.Dense(1)(critic)
        
        return action_mean, action_logstd, value


def create_train_state(rng, obs_dim, act_dim, hidden_sizes, learning_rate):
    """Create Flax TrainState."""
    network = PPONetwork(hidden_sizes=hidden_sizes, act_dim=act_dim)
    params = network.init(rng, jnp.zeros((1, obs_dim)))
    tx = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(learning_rate, eps=1e-5),
    )
    return TrainState.create(apply_fn=network.apply, params=params, tx=tx)


# =============================================================================
# PPO Functions
# =============================================================================

class Transition(NamedTuple):
    """Single transition for PPO."""
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    done: jax.Array
    value: jax.Array
    log_prob: jax.Array


def sample_action(rng, train_state, obs):
    """Sample action from policy."""
    action_mean, action_logstd, value = train_state.apply_fn(train_state.params, obs)
    action_std = jnp.exp(action_logstd)
    action = action_mean + action_std * jax.random.normal(rng, action_mean.shape)
    log_prob = -0.5 * jnp.sum(
        jnp.square((action - action_mean) / action_std) + 2 * action_logstd + jnp.log(2 * jnp.pi),
        axis=-1
    )
    return action, log_prob, value.squeeze(-1)


def compute_gae(rewards, values, dones, last_value, gamma, gae_lambda):
    """Compute Generalized Advantage Estimation."""
    def scan_fn(carry, transition):
        gae, next_value = carry
        reward, value, done = transition
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * gae_lambda * (1 - done) * gae
        return (gae, value), gae
    
    _, advantages = jax.lax.scan(
        scan_fn,
        (jnp.zeros_like(last_value), last_value),
        (rewards[::-1], values[::-1], dones[::-1]),
    )
    advantages = advantages[::-1]
    returns = advantages + values
    return advantages, returns


def ppo_loss(params, apply_fn, batch, clip_coef, vf_coef, ent_coef):
    """Compute PPO loss."""
    obs, actions, old_log_probs, advantages, returns, old_values = batch
    
    action_mean, action_logstd, values = apply_fn(params, obs)
    action_std = jnp.exp(action_logstd)
    values = values.squeeze(-1)
    
    # Log prob of actions under new policy
    log_probs = -0.5 * jnp.sum(
        jnp.square((actions - action_mean) / action_std) + 2 * action_logstd + jnp.log(2 * jnp.pi),
        axis=-1
    )
    
    # Policy loss
    ratio = jnp.exp(log_probs - old_log_probs)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * jnp.clip(ratio, 1 - clip_coef, 1 + clip_coef)
    pg_loss = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))
    
    # Value loss
    v_loss = 0.5 * jnp.mean(jnp.square(values - returns))
    
    # Entropy
    entropy = 0.5 * jnp.sum(1 + jnp.log(2 * jnp.pi) + 2 * action_logstd)
    
    total_loss = pg_loss + vf_coef * v_loss - ent_coef * entropy
    return total_loss, (pg_loss, v_loss, entropy)


@functools.partial(jax.jit, static_argnums=(3, 4, 5, 6))
def ppo_update(train_state, batch, rng, num_epochs, minibatch_size, clip_coef, vf_coef, ent_coef):
    """Run PPO update epochs."""
    batch_size = batch[0].shape[0]
    
    def epoch_step(carry, _):
        train_state, rng = carry
        rng, perm_rng = jax.random.split(rng)
        perm = jax.random.permutation(perm_rng, batch_size)
        
        def minibatch_step(train_state, start_idx):
            idx = jax.lax.dynamic_slice(perm, (start_idx,), (minibatch_size,))
            mb = tuple(x[idx] for x in batch)
            
            grads, _ = jax.grad(ppo_loss, has_aux=True)(
                train_state.params, train_state.apply_fn, mb, clip_coef, vf_coef, ent_coef
            )
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, None
        
        num_minibatches = batch_size // minibatch_size
        train_state, _ = jax.lax.scan(
            minibatch_step,
            train_state,
            jnp.arange(0, batch_size, minibatch_size)[:num_minibatches],
        )
        return (train_state, rng), None
    
    (train_state, _), _ = jax.lax.scan(epoch_step, (train_state, rng), None, length=num_epochs)
    return train_state


# =============================================================================
# Training Loop
# =============================================================================

def train(config: MJXConfig, checkpoint_dir: str, experiment_name: str):
    """Main MJX training loop."""
    
    # Setup
    rng = jax.random.PRNGKey(config.seed)
    model_path = Path(__file__).parent.parent / "humanoid_hug" / "mjcf" / "humanoid_hug.xml"
    
    print(f"\nInitializing MJX environment with {config.num_envs} parallel envs...")
    env = HumanoidHugMJX(
        str(model_path),
        num_envs=config.num_envs,
        horizon=config.horizon,
        frame_skip=config.frame_skip,
        stage=config.stage,
    )
    
    # Create agents
    rng, rng_h0, rng_h1 = jax.random.split(rng, 3)
    h0_state = create_train_state(rng_h0, env.obs_dim, env.act_dim, config.hidden_sizes, config.learning_rate)
    h1_state = create_train_state(rng_h1, env.obs_dim, env.act_dim, config.hidden_sizes, config.learning_rate)
    
    # Logging
    run_name = f"{experiment_name}_{config.seed}_{int(time.time())}"
    log_dir = Path(checkpoint_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Logging to: {log_dir}")
    print(f"\nStarting training for {config.total_timesteps:,} timesteps ({config.num_updates} updates)")
    print(f"Batch size: {config.batch_size:,} ({config.num_envs} envs × {config.num_steps} steps)")
    print("=" * 60)
    
    # Initial reset
    rng, rng_reset = jax.random.split(rng)
    env_state, obs = env.reset(rng_reset)
    
    # Tracking
    episode_returns = []
    episode_lengths = []
    global_step = 0
    start_time = time.time()
    
    for update in range(1, config.num_updates + 1):
        update_start = time.time()
        
        # Collect rollout
        h0_transitions = []
        h1_transitions = []
        
        for step in range(config.num_steps):
            rng, rng_h0_act, rng_h1_act = jax.random.split(rng, 3)
            
            # Sample actions
            h0_action, h0_log_prob, h0_value = sample_action(rng_h0_act, h0_state, obs["h0"])
            h1_action, h1_log_prob, h1_value = sample_action(rng_h1_act, h1_state, obs["h1"])
            
            # Step environment
            env_state, next_obs, rewards, dones, info = env.step(env_state, h0_action, h1_action)
            
            # Store transitions
            h0_transitions.append(Transition(obs["h0"], h0_action, rewards, dones, h0_value, h0_log_prob))
            h1_transitions.append(Transition(obs["h1"], h1_action, rewards, dones, h1_value, h1_log_prob))
            
            obs = next_obs
            global_step += config.num_envs
            
            # Track episodes
            successes = info["success"].sum().item()
            if successes > 0:
                episode_returns.extend([100.0] * int(successes))
                episode_lengths.extend([config.horizon] * int(successes))
        
        # Stack transitions
        def stack_transitions(transitions):
            return Transition(*[jnp.stack([t[i] for t in transitions]) for i in range(6)])
        
        h0_batch = stack_transitions(h0_transitions)
        h1_batch = stack_transitions(h1_transitions)
        
        # Compute advantages
        rng, rng_h0_val, rng_h1_val = jax.random.split(rng, 3)
        _, _, h0_last_value = sample_action(rng_h0_val, h0_state, obs["h0"])
        _, _, h1_last_value = sample_action(rng_h1_val, h1_state, obs["h1"])
        
        h0_advantages, h0_returns = compute_gae(
            h0_batch.reward, h0_batch.value, h0_batch.done, h0_last_value, config.gamma, config.gae_lambda
        )
        h1_advantages, h1_returns = compute_gae(
            h1_batch.reward, h1_batch.value, h1_batch.done, h1_last_value, config.gamma, config.gae_lambda
        )
        
        # Flatten and normalize advantages
        def prepare_batch(batch, advantages, returns):
            obs_flat = batch.obs.reshape(-1, batch.obs.shape[-1])
            actions_flat = batch.action.reshape(-1, batch.action.shape[-1])
            log_probs_flat = batch.log_prob.reshape(-1)
            values_flat = batch.value.reshape(-1)
            advantages_flat = advantages.reshape(-1)
            returns_flat = returns.reshape(-1)
            advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)
            return (obs_flat, actions_flat, log_probs_flat, advantages_flat, returns_flat, values_flat)
        
        h0_ppo_batch = prepare_batch(h0_batch, h0_advantages, h0_returns)
        h1_ppo_batch = prepare_batch(h1_batch, h1_advantages, h1_returns)
        
        # PPO updates
        rng, rng_h0_update, rng_h1_update = jax.random.split(rng, 3)
        h0_state = ppo_update(
            h0_state, h0_ppo_batch, rng_h0_update,
            config.num_epochs, config.minibatch_size, config.clip_coef, config.vf_coef, config.ent_coef
        )
        h1_state = ppo_update(
            h1_state, h1_ppo_batch, rng_h1_update,
            config.num_epochs, config.minibatch_size, config.clip_coef, config.vf_coef, config.ent_coef
        )
        
        update_time = time.time() - update_start
        fps = config.batch_size / update_time
        
        # Logging
        if update % config.log_interval == 0:
            elapsed = time.time() - start_time
            mean_reward = jnp.mean(h0_batch.reward).item()
            
            print(f"Update {update:4d}/{config.num_updates} | "
                  f"Steps: {global_step:,} | "
                  f"FPS: {fps:,.0f} | "
                  f"Reward: {mean_reward:.2f} | "
                  f"Time: {elapsed:.0f}s")
        
        # Save checkpoint
        if update % config.save_interval == 0:
            checkpoint_path = log_dir / f"checkpoint_{update}.npz"
            jnp.savez(
                checkpoint_path,
                h0_params=h0_state.params,
                h1_params=h1_state.params,
                update=update,
                global_step=global_step,
            )
            print(f"  Saved checkpoint: {checkpoint_path}")
    
    # Final save
    final_path = log_dir / "final_model.npz"
    jnp.savez(
        final_path,
        h0_params=h0_state.params,
        h1_params=h1_state.params,
    )
    
    total_time = time.time() - start_time
    print(f"\nTraining complete!")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Average FPS: {config.total_timesteps / total_time:,.0f}")
    print(f"Final model saved to: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="MJX-accelerated PPO training for Humanoid Hug")
    
    # Environment
    parser.add_argument("--stage", type=int, default=0, help="Curriculum stage (0-3)")
    parser.add_argument("--horizon", type=int, default=1000, help="Episode horizon")
    parser.add_argument("--frame-skip", type=int, default=5, help="Physics steps per action")
    
    # Training
    parser.add_argument("--total-timesteps", type=int, default=20_000_000, help="Total timesteps")
    parser.add_argument("--num-envs", type=int, default=4096, help="Number of parallel GPU environments")
    parser.add_argument("--num-steps", type=int, default=64, help="Rollout steps per update")
    parser.add_argument("--num-epochs", type=int, default=4, help="PPO epochs per update")
    parser.add_argument("--minibatch-size", type=int, default=4096, help="Minibatch size")
    
    # PPO
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="PPO clip coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")
    
    # Network
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256], help="Hidden layer sizes")
    
    # Logging
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints_mjx", help="Checkpoint directory")
    parser.add_argument("--experiment-name", type=str, default="humanoid_hug_mjx", help="Experiment name")
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N updates")
    parser.add_argument("--save-interval", type=int, default=100, help="Save every N updates")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    config = MJXConfig(
        horizon=args.horizon,
        frame_skip=args.frame_skip,
        stage=args.stage,
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        num_epochs=args.num_epochs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        hidden_sizes=tuple(args.hidden_sizes),
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        seed=args.seed,
    )
    
    print("=" * 60)
    print("Humanoid Hug - MJX GPU Training")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Stage: {config.stage}")
    print(f"Total timesteps: {config.total_timesteps:,}")
    print(f"Parallel GPU envs: {config.num_envs:,}")
    print(f"Batch size: {config.batch_size:,}")
    print(f"Network: {config.hidden_sizes}")
    print("=" * 60)
    
    train(config, args.checkpoint_dir, args.experiment_name)


if __name__ == "__main__":
    main()
