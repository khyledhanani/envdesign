#!/usr/bin/env python3
"""
MJX-accelerated IPPO training for Humanoid Hug environment.

Runs thousands of environments in parallel on GPU using MuJoCo XLA (MJX).
Achieves 50-100x speedup over CPU-based training.

Requirements:
    pip install mujoco-mjx jax[cuda12] flax optax

Usage:
    python scripts/train_mjx.py --num-envs 2048 --total-timesteps 20_000_000
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
from tqdm import tqdm

# Check JAX backend
print(f"JAX devices: {jax.devices()}")
print(f"JAX default backend: {jax.default_backend()}")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MJXConfig:
    """Configuration for MJX PPO training."""
    horizon: int = 1000
    frame_skip: int = 5
    stage: int = 0
    total_timesteps: int = 20_000_000
    num_envs: int = 2048
    num_steps: int = 64
    num_epochs: int = 4
    minibatch_size: int = 4096
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_sizes: Tuple[int, ...] = (256, 256)
    log_interval: int = 10
    save_interval: int = 100
    seed: int = 42
    
    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps
    
    @property
    def num_updates(self) -> int:
        return self.total_timesteps // self.batch_size


# =============================================================================
# Environment State
# =============================================================================

class EnvState(NamedTuple):
    """Batched environment state."""
    data: mjx.Data          # Batched MJX data (num_envs, ...)
    step_count: jax.Array   # (num_envs,)
    hug_hold: jax.Array     # (num_envs,)
    rng: jax.Array


class EnvParams(NamedTuple):
    """Static environment parameters."""
    mjx_model: mjx.Model
    h0_act_idx: jax.Array
    h1_act_idx: jax.Array
    h0_torso_idx: int
    h1_torso_idx: int
    h0_qpos_start: int
    h1_qpos_start: int
    h0_qvel_start: int
    h1_qvel_start: int
    nq: int
    nv: int
    nu: int
    frame_skip: int
    horizon: int
    weights: Dict[str, float]


# =============================================================================
# Pure Functions (JIT-friendly)
# =============================================================================

def make_env_params(model_path: str, frame_skip: int, horizon: int, stage: int) -> Tuple[EnvParams, int, int]:
    """Create environment parameters from model file."""
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mjx_model = mjx.put_model(mj_model)
    
    # Find actuator indices
    h0_act, h1_act = [], []
    for i in range(mj_model.nu):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name and name.startswith("h0_"):
            h0_act.append(i)
        elif name and name.startswith("h1_"):
            h1_act.append(i)
    
    # Find body indices
    h0_torso = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "h0_torso")
    h1_torso = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "h1_torso")
    
    # Find joint qpos/qvel starts
    h0_root = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "h0_root")
    h1_root = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "h1_root")
    h0_qpos_start = mj_model.jnt_qposadr[h0_root]
    h1_qpos_start = mj_model.jnt_qposadr[h1_root]
    h0_qvel_start = mj_model.jnt_dofadr[h0_root]
    h1_qvel_start = mj_model.jnt_dofadr[h1_root]
    
    # Reward weights by stage
    stage_weights = {
        0: {"dist": 2.0, "facing": 1.0, "stability": 0.5, "contact": 0.0, "success": 100.0},
        1: {"dist": 1.5, "facing": 1.0, "stability": 0.5, "contact": 0.0, "success": 100.0},
        2: {"dist": 1.0, "facing": 0.8, "stability": 0.5, "contact": 3.0, "success": 150.0},
        3: {"dist": 0.5, "facing": 0.5, "stability": 0.3, "contact": 5.0, "success": 200.0},
    }
    
    params = EnvParams(
        mjx_model=mjx_model,
        h0_act_idx=jnp.array(h0_act),
        h1_act_idx=jnp.array(h1_act),
        h0_torso_idx=h0_torso,
        h1_torso_idx=h1_torso,
        h0_qpos_start=h0_qpos_start,
        h1_qpos_start=h1_qpos_start,
        h0_qvel_start=h0_qvel_start,
        h1_qvel_start=h1_qvel_start,
        nq=mj_model.nq // 2,  # per humanoid
        nv=mj_model.nv // 2,
        nu=mj_model.nu,
        frame_skip=frame_skip,
        horizon=horizon,
        weights=stage_weights.get(stage, stage_weights[0]),
    )
    
    obs_dim = 68  # Fixed for this env
    act_dim = len(h0_act)
    
    return params, obs_dim, act_dim


@functools.partial(jax.jit, static_argnums=(1, 2))
def reset_env(rng: jax.Array, params: EnvParams, num_envs: int) -> EnvState:
    """Reset all environments."""
    rngs = jax.random.split(rng, num_envs + 1)
    rng, reset_rngs = rngs[0], rngs[1:]
    
    data = jax.vmap(lambda r: _reset_single(r, params))(reset_rngs)
    
    return EnvState(
        data=data,
        step_count=jnp.zeros(num_envs, dtype=jnp.int32),
        hug_hold=jnp.zeros(num_envs, dtype=jnp.int32),
        rng=rng,
    )


def _reset_single(rng: jax.Array, params: EnvParams) -> mjx.Data:
    """Reset a single environment."""
    rng, r1, r2, r3, r4, r5 = jax.random.split(rng, 6)
    
    data = mjx.make_data(params.mjx_model)
    qpos = data.qpos
    qvel = jnp.zeros_like(data.qvel)
    
    # Random initial distance
    half_dist = jax.random.uniform(r1, minval=0.75, maxval=1.25)
    
    # H0
    h0_x = -half_dist + jax.random.uniform(r2, minval=-0.1, maxval=0.1)
    h0_y = jax.random.uniform(r2, minval=-0.1, maxval=0.1)
    yaw0 = jax.random.uniform(r3, minval=-0.2, maxval=0.2)
    
    s = params.h0_qpos_start
    qpos = qpos.at[s:s+3].set(jnp.array([h0_x, h0_y, 1.4]))
    qpos = qpos.at[s+3:s+7].set(jnp.array([jnp.cos(yaw0/2), 0, 0, jnp.sin(yaw0/2)]))
    
    # H1
    h1_x = half_dist + jax.random.uniform(r4, minval=-0.1, maxval=0.1)
    h1_y = jax.random.uniform(r4, minval=-0.1, maxval=0.1)
    yaw1 = jnp.pi + jax.random.uniform(r5, minval=-0.2, maxval=0.2)
    
    s = params.h1_qpos_start
    qpos = qpos.at[s:s+3].set(jnp.array([h1_x, h1_y, 1.4]))
    qpos = qpos.at[s+3:s+7].set(jnp.array([jnp.cos(yaw1/2), 0, 0, jnp.sin(yaw1/2)]))
    
    data = data.replace(qpos=qpos, qvel=qvel)
    return mjx.forward(params.mjx_model, data)


@functools.partial(jax.jit, static_argnums=(3,))
def step_env(
    state: EnvState,
    h0_act: jax.Array,
    h1_act: jax.Array,
    params: EnvParams,
) -> Tuple[EnvState, Dict[str, jax.Array], jax.Array, jax.Array, Dict]:
    """Step all environments."""
    num_envs = h0_act.shape[0]
    
    # Build control
    ctrl = jnp.zeros((num_envs, params.nu))
    ctrl = ctrl.at[:, params.h0_act_idx].set(jnp.clip(h0_act, -1, 1))
    ctrl = ctrl.at[:, params.h1_act_idx].set(jnp.clip(h1_act, -1, 1))
    
    # Step physics
    data = state.data.replace(ctrl=ctrl)
    data = jax.vmap(lambda d: _physics_step(d, params))(data)
    
    # Observations
    obs = _get_obs_batched(data, params)
    
    # Hug check
    hug_ok = jax.vmap(lambda d: _check_hug(d, params))(data)
    new_hug_hold = jnp.where(hug_ok, state.hug_hold + 1, 0)
    
    # Fallen check
    fallen = jax.vmap(lambda d: _check_fallen(d, params))(data)
    
    # Success
    success = new_hug_hold >= 30
    
    # Rewards
    rewards = jax.vmap(lambda d, c, s, f: _compute_reward(d, c, s, f, params))(
        data, ctrl, success, fallen
    )
    
    # Done
    new_step = state.step_count + 1
    done = fallen | success | (new_step >= params.horizon)
    
    # Auto-reset
    rng, reset_rng = jax.random.split(state.rng)
    reset_rngs = jax.random.split(reset_rng, num_envs)
    
    def maybe_reset(d, reset_flag, r):
        new_d = _reset_single(r, params)
        return jax.lax.select(reset_flag, new_d, d)
    
    data = jax.vmap(maybe_reset)(data, done, reset_rngs)
    new_step = jnp.where(done, 0, new_step)
    new_hug_hold = jnp.where(done, 0, new_hug_hold)
    
    new_state = EnvState(data=data, step_count=new_step, hug_hold=new_hug_hold, rng=rng)
    info = {"success": success, "fallen": fallen}
    
    return new_state, obs, rewards, done, info


def _physics_step(data: mjx.Data, params: EnvParams) -> mjx.Data:
    """Step physics for one env."""
    def do_step(d, _):
        return mjx.step(params.mjx_model, d), None
    d, _ = jax.lax.scan(do_step, data, None, length=params.frame_skip)
    return d


def _get_obs_batched(data: mjx.Data, params: EnvParams) -> Dict[str, jax.Array]:
    """Get observations for all envs."""
    h0_obs = jax.vmap(lambda d: _get_agent_obs(d, params, is_h0=True))(data)
    h1_obs = jax.vmap(lambda d: _get_agent_obs(d, params, is_h0=False))(data)
    return {"h0": h0_obs, "h1": h1_obs}


def _get_agent_obs(data: mjx.Data, params: EnvParams, is_h0: bool) -> jax.Array:
    """Get observation for one agent."""
    if is_h0:
        qpos_s, qvel_s = params.h0_qpos_start, params.h0_qvel_start
        torso_idx, partner_idx = params.h0_torso_idx, params.h1_torso_idx
        p_qvel_s = params.h1_qvel_start
    else:
        qpos_s, qvel_s = params.h1_qpos_start, params.h1_qvel_start
        torso_idx, partner_idx = params.h1_torso_idx, params.h0_torso_idx
        p_qvel_s = params.h0_qvel_start
    
    nq, nv = params.nq, params.nv
    
    # Proprioception
    joint_qpos = data.qpos[qpos_s + 3: qpos_s + nq]
    joint_qvel = data.qvel[qvel_s + 3: qvel_s + nv]
    root_quat = data.qpos[qpos_s + 3: qpos_s + 7]
    root_angvel = data.qvel[qvel_s + 3: qvel_s + 6]
    
    # Partner relative
    self_pos = data.xpos[torso_idx]
    self_mat = data.xmat[torso_idx].reshape(3, 3)
    partner_pos = data.xpos[partner_idx]
    partner_mat = data.xmat[partner_idx].reshape(3, 3)
    
    rel_pos = self_mat.T @ (partner_pos - self_pos)
    rel_pelvis = self_mat.T @ (partner_pos - jnp.array([0, 0, 0.2]) - self_pos)
    
    self_vel = data.qvel[qvel_s: qvel_s + 3]
    partner_vel = data.qvel[p_qvel_s: p_qvel_s + 3]
    rel_vel = self_mat.T @ (partner_vel - self_vel)
    
    facing = -jnp.dot(self_mat[:, 0], partner_mat[:, 0])
    up_align = jnp.dot(self_mat[:, 2], partner_mat[:, 2])
    
    dist = jnp.linalg.norm(partner_pos - self_pos)
    contact_proxy = (dist < 0.6).astype(jnp.float32)
    
    return jnp.concatenate([
        joint_qpos, joint_qvel, root_quat, root_angvel,
        rel_pos, rel_pelvis, rel_vel,
        jnp.array([facing, up_align]),
        jnp.array([contact_proxy, contact_proxy, contact_proxy]),
    ])


def _check_hug(data: mjx.Data, params: EnvParams) -> jax.Array:
    """Check hug condition."""
    h0_pos = data.xpos[params.h0_torso_idx]
    h1_pos = data.xpos[params.h1_torso_idx]
    h0_mat = data.xmat[params.h0_torso_idx].reshape(3, 3)
    h1_mat = data.xmat[params.h1_torso_idx].reshape(3, 3)
    
    dist = jnp.linalg.norm(h0_pos - h1_pos)
    facing = -jnp.dot(h0_mat[:, 0], h1_mat[:, 0])
    
    h0_vel = data.qvel[params.h0_qvel_start: params.h0_qvel_start + 3]
    h1_vel = data.qvel[params.h1_qvel_start: params.h1_qvel_start + 3]
    rel_speed = jnp.linalg.norm(h0_vel - h1_vel)
    
    h0_tilt = jnp.arccos(jnp.clip(h0_mat[2, 2], -1, 1))
    h1_tilt = jnp.arccos(jnp.clip(h1_mat[2, 2], -1, 1))
    
    return (
        (dist > 0.25) & (dist < 0.6) &
        (facing > 0.6) &
        (rel_speed < 1.0) &
        (h0_tilt < 0.5) & (h1_tilt < 0.5)
    )


def _check_fallen(data: mjx.Data, params: EnvParams) -> jax.Array:
    """Check if fallen."""
    h0_z = data.xpos[params.h0_torso_idx, 2]
    h1_z = data.xpos[params.h1_torso_idx, 2]
    h0_tilt = jnp.arccos(jnp.clip(data.xmat[params.h0_torso_idx].reshape(3, 3)[2, 2], -1, 1))
    h1_tilt = jnp.arccos(jnp.clip(data.xmat[params.h1_torso_idx].reshape(3, 3)[2, 2], -1, 1))
    return (h0_z < 0.5) | (h1_z < 0.5) | (h0_tilt > jnp.pi/2) | (h1_tilt > jnp.pi/2)


def _compute_reward(data: mjx.Data, ctrl: jax.Array, success: jax.Array, fallen: jax.Array, params: EnvParams) -> jax.Array:
    """Compute reward."""
    w = params.weights
    h0_pos = data.xpos[params.h0_torso_idx]
    h1_pos = data.xpos[params.h1_torso_idx]
    h0_mat = data.xmat[params.h0_torso_idx].reshape(3, 3)
    h1_mat = data.xmat[params.h1_torso_idx].reshape(3, 3)
    
    dist = jnp.linalg.norm(h0_pos - h1_pos)
    facing = -jnp.dot(h0_mat[:, 0], h1_mat[:, 0])
    
    h0_vel = data.qvel[params.h0_qvel_start: params.h0_qvel_start + 3]
    h1_vel = data.qvel[params.h1_qvel_start: params.h1_qvel_start + 3]
    rel_speed = jnp.linalg.norm(h0_vel - h1_vel)
    
    r = (
        w["dist"] * jnp.exp(-3.0 * dist) +
        w["facing"] * jnp.maximum(0.0, facing) +
        w["stability"] * jnp.exp(-2.0 * rel_speed) +
        w["contact"] * (dist < 0.55).astype(jnp.float32) * 2 +
        -0.001 * jnp.sum(jnp.square(ctrl)) +
        jnp.where(fallen, -100.0, 0.0) +
        jnp.where(success, w["success"], 0.0)
    )
    return r


# =============================================================================
# Neural Network
# =============================================================================

class PPONetwork(nn.Module):
    hidden_sizes: Tuple[int, ...]
    act_dim: int
    
    @nn.compact
    def __call__(self, x):
        # Actor
        a = x
        for sz in self.hidden_sizes:
            a = nn.tanh(nn.Dense(sz)(a))
        mean = nn.Dense(self.act_dim)(a)
        logstd = self.param("logstd", nn.initializers.zeros, (self.act_dim,))
        
        # Critic
        c = x
        for sz in self.hidden_sizes:
            c = nn.tanh(nn.Dense(sz)(c))
        value = nn.Dense(1)(c)
        
        return mean, logstd, value


def create_train_state(rng, obs_dim, act_dim, hidden_sizes, lr):
    net = PPONetwork(hidden_sizes=hidden_sizes, act_dim=act_dim)
    params = net.init(rng, jnp.zeros((1, obs_dim)))
    tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr, eps=1e-5))
    return TrainState.create(apply_fn=net.apply, params=params, tx=tx)


# =============================================================================
# PPO
# =============================================================================

class Transition(NamedTuple):
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    done: jax.Array
    value: jax.Array
    log_prob: jax.Array


@jax.jit
def sample_action(rng, state, obs):
    mean, logstd, value = state.apply_fn(state.params, obs)
    std = jnp.exp(logstd)
    action = mean + std * jax.random.normal(rng, mean.shape)
    log_prob = -0.5 * jnp.sum(jnp.square((action - mean) / std) + 2 * logstd + jnp.log(2 * jnp.pi), axis=-1)
    return action, log_prob, value.squeeze(-1)


@jax.jit
def compute_gae(rewards, values, dones, last_val, gamma, lam):
    def scan_fn(carry, t):
        gae, next_v = carry
        r, v, d = t
        delta = r + gamma * next_v * (1 - d) - v
        gae = delta + gamma * lam * (1 - d) * gae
        return (gae, v), gae
    
    _, advs = jax.lax.scan(scan_fn, (jnp.zeros_like(last_val), last_val),
                           (rewards[::-1], values[::-1], dones[::-1]))
    return advs[::-1], advs[::-1] + values


@functools.partial(jax.jit, static_argnums=(3, 4))
def ppo_update(state, batch, rng, num_epochs, minibatch_size, clip_coef=0.2, vf_coef=0.5, ent_coef=0.01):
    obs, actions, old_log_probs, advantages, returns, _ = batch
    batch_size = obs.shape[0]
    
    def loss_fn(params, mb_obs, mb_act, mb_old_lp, mb_adv, mb_ret):
        mean, logstd, value = state.apply_fn(params, mb_obs)
        std = jnp.exp(logstd)
        value = value.squeeze(-1)
        
        log_prob = -0.5 * jnp.sum(jnp.square((mb_act - mean) / std) + 2 * logstd + jnp.log(2 * jnp.pi), axis=-1)
        ratio = jnp.exp(log_prob - mb_old_lp)
        
        pg1 = -mb_adv * ratio
        pg2 = -mb_adv * jnp.clip(ratio, 1 - clip_coef, 1 + clip_coef)
        pg_loss = jnp.mean(jnp.maximum(pg1, pg2))
        
        v_loss = 0.5 * jnp.mean(jnp.square(value - mb_ret))
        entropy = 0.5 * jnp.sum(1 + jnp.log(2 * jnp.pi) + 2 * logstd)
        
        return pg_loss + vf_coef * v_loss - ent_coef * entropy
    
    def epoch_step(carry, _):
        state, rng = carry
        rng, perm_rng = jax.random.split(rng)
        perm = jax.random.permutation(perm_rng, batch_size)
        
        def mb_step(state, start):
            idx = jax.lax.dynamic_slice(perm, (start,), (minibatch_size,))
            grads = jax.grad(loss_fn)(state.params, obs[idx], actions[idx], 
                                       old_log_probs[idx], advantages[idx], returns[idx])
            return state.apply_gradients(grads=grads), None
        
        n_mb = batch_size // minibatch_size
        state, _ = jax.lax.scan(mb_step, state, jnp.arange(0, batch_size, minibatch_size)[:n_mb])
        return (state, rng), None
    
    (state, _), _ = jax.lax.scan(epoch_step, (state, rng), None, length=num_epochs)
    return state


# =============================================================================
# Training
# =============================================================================

def train(config: MJXConfig, checkpoint_dir: str, experiment_name: str):
    rng = jax.random.PRNGKey(config.seed)
    model_path = str(Path(__file__).parent.parent / "humanoid_hug" / "mjcf" / "humanoid_hug.xml")
    
    print(f"\nInitializing MJX environment with {config.num_envs} parallel envs...")
    params, obs_dim, act_dim = make_env_params(model_path, config.frame_skip, config.horizon, config.stage)
    print(f"  obs_dim={obs_dim}, act_dim={act_dim}")
    
    rng, rng_h0, rng_h1 = jax.random.split(rng, 3)
    h0_state = create_train_state(rng_h0, obs_dim, act_dim, config.hidden_sizes, config.learning_rate)
    h1_state = create_train_state(rng_h1, obs_dim, act_dim, config.hidden_sizes, config.learning_rate)
    
    run_name = f"{experiment_name}_{config.seed}_{int(time.time())}"
    log_dir = Path(checkpoint_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Logging to: {log_dir}")
    print(f"\nStarting training for {config.total_timesteps:,} timesteps ({config.num_updates} updates)")
    print(f"Batch size: {config.batch_size:,} ({config.num_envs} envs × {config.num_steps} steps)")
    print("=" * 60)
    print("\nJIT compiling (first step is slow, please wait)...")
    
    rng, rng_reset = jax.random.split(rng)
    env_state = reset_env(rng_reset, params, config.num_envs)
    obs = _get_obs_batched(env_state.data, params)
    
    global_step = 0
    start_time = time.time()
    pbar = tqdm(range(1, config.num_updates + 1), desc="Training", unit="update")
    
    for update in pbar:
        update_start = time.time()
        
        # Collect rollout
        h0_trans, h1_trans = [], []
        
        for _ in range(config.num_steps):
            rng, r1, r2 = jax.random.split(rng, 3)
            
            h0_act, h0_lp, h0_val = sample_action(r1, h0_state, obs["h0"])
            h1_act, h1_lp, h1_val = sample_action(r2, h1_state, obs["h1"])
            
            env_state, next_obs, rewards, dones, _ = step_env(env_state, h0_act, h1_act, params)
            
            h0_trans.append(Transition(obs["h0"], h0_act, rewards, dones, h0_val, h0_lp))
            h1_trans.append(Transition(obs["h1"], h1_act, rewards, dones, h1_val, h1_lp))
            
            obs = next_obs
            global_step += config.num_envs
        
        # Stack
        h0_batch = Transition(*[jnp.stack([t[i] for t in h0_trans]) for i in range(6)])
        h1_batch = Transition(*[jnp.stack([t[i] for t in h1_trans]) for i in range(6)])
        
        # GAE
        rng, r1, r2 = jax.random.split(rng, 3)
        _, _, h0_last = sample_action(r1, h0_state, obs["h0"])
        _, _, h1_last = sample_action(r2, h1_state, obs["h1"])
        
        h0_adv, h0_ret = compute_gae(h0_batch.reward, h0_batch.value, h0_batch.done, h0_last, config.gamma, config.gae_lambda)
        h1_adv, h1_ret = compute_gae(h1_batch.reward, h1_batch.value, h1_batch.done, h1_last, config.gamma, config.gae_lambda)
        
        # Flatten
        def flatten_batch(batch, adv, ret):
            o = batch.obs.reshape(-1, batch.obs.shape[-1])
            a = batch.action.reshape(-1, batch.action.shape[-1])
            lp = batch.log_prob.reshape(-1)
            v = batch.value.reshape(-1)
            adv_f = adv.reshape(-1)
            adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)
            ret_f = ret.reshape(-1)
            return (o, a, lp, adv_f, ret_f, v)
        
        h0_ppo = flatten_batch(h0_batch, h0_adv, h0_ret)
        h1_ppo = flatten_batch(h1_batch, h1_adv, h1_ret)
        
        # Update
        rng, r1, r2 = jax.random.split(rng, 3)
        h0_state = ppo_update(h0_state, h0_ppo, r1, config.num_epochs, config.minibatch_size)
        h1_state = ppo_update(h1_state, h1_ppo, r2, config.num_epochs, config.minibatch_size)
        
        fps = config.batch_size / (time.time() - update_start)
        mean_rew = float(jnp.mean(h0_batch.reward))
        
        pbar.set_postfix({"reward": f"{mean_rew:.2f}", "fps": f"{fps:,.0f}", "steps": f"{global_step:,}"})
        
        if update % config.save_interval == 0:
            ckpt = log_dir / f"checkpoint_{update}.npz"
            jnp.savez(ckpt, h0=h0_state.params, h1=h1_state.params, update=update)
            tqdm.write(f"  Saved: {ckpt}")
    
    pbar.close()
    final = log_dir / "final_model.npz"
    jnp.savez(final, h0=h0_state.params, h1=h1_state.params)
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training complete! Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Average FPS: {config.total_timesteps / total_time:,.0f}")
    print(f"Final model: {final}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="MJX PPO Training")
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument("--total-timesteps", type=int, default=20_000_000)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints_mjx")
    parser.add_argument("--experiment-name", type=str, default="humanoid_hug_mjx")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    config = MJXConfig(
        horizon=args.horizon, frame_skip=args.frame_skip, stage=args.stage,
        total_timesteps=args.total_timesteps, num_envs=args.num_envs,
        num_steps=args.num_steps, num_epochs=args.num_epochs,
        minibatch_size=args.minibatch_size, learning_rate=args.lr,
        gamma=args.gamma, gae_lambda=args.gae_lambda, clip_coef=args.clip_coef,
        ent_coef=args.ent_coef, vf_coef=args.vf_coef,
        hidden_sizes=tuple(args.hidden_sizes),
        log_interval=args.log_interval, save_interval=args.save_interval,
        seed=args.seed,
    )
    
    print("=" * 60)
    print("Humanoid Hug - MJX GPU Training")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Stage: {config.stage}, Envs: {config.num_envs:,}, Batch: {config.batch_size:,}")
    print("=" * 60)
    
    train(config, args.checkpoint_dir, args.experiment_name)


if __name__ == "__main__":
    main()
