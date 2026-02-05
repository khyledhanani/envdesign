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
# Global Environment State
# =============================================================================

_MJX_MODEL: Optional[mjx.Model] = None
_H0_ACT_IDX: Optional[jax.Array] = None
_H1_ACT_IDX: Optional[jax.Array] = None
_H0_TORSO_IDX: int = 0
_H1_TORSO_IDX: int = 0
_H0_QPOS_START: int = 0
_H1_QPOS_START: int = 0
_H0_QVEL_START: int = 0
_H1_QVEL_START: int = 0
_NQ: int = 27
_NV: int = 26
_NU: int = 36
_FRAME_SKIP: int = 5
_HORIZON: int = 1000
_WEIGHTS: Dict[str, float] = {}


def init_env(model_path: str, frame_skip: int, horizon: int, stage: int) -> Tuple[int, int]:
    global _MJX_MODEL, _H0_ACT_IDX, _H1_ACT_IDX
    global _H0_TORSO_IDX, _H1_TORSO_IDX
    global _H0_QPOS_START, _H1_QPOS_START, _H0_QVEL_START, _H1_QVEL_START
    global _NQ, _NV, _NU, _FRAME_SKIP, _HORIZON, _WEIGHTS
    
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    _MJX_MODEL = mjx.put_model(mj_model)
    
    h0_act, h1_act = [], []
    for i in range(mj_model.nu):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name and name.startswith("h0_"): h0_act.append(i)
        elif name and name.startswith("h1_"): h1_act.append(i)
    _H0_ACT_IDX = jnp.array(h0_act)
    _H1_ACT_IDX = jnp.array(h1_act)
    
    _H0_TORSO_IDX = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "h0_torso")
    _H1_TORSO_IDX = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "h1_torso")
    
    h0_root = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "h0_root")
    h1_root = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "h1_root")
    _H0_QPOS_START = int(mj_model.jnt_qposadr[h0_root])
    _H1_QPOS_START = int(mj_model.jnt_qposadr[h1_root])
    _H0_QVEL_START = int(mj_model.jnt_dofadr[h0_root])
    _H1_QVEL_START = int(mj_model.jnt_dofadr[h1_root])
    
    _NQ = mj_model.nq // 2
    _NV = mj_model.nv // 2
    _NU = mj_model.nu
    _FRAME_SKIP = frame_skip
    _HORIZON = horizon
    
    stage_weights = {
        0: {"dist": 2.0, "facing": 1.0, "stability": 0.5, "contact": 0.0, "success": 100.0},
        1: {"dist": 1.5, "facing": 1.0, "stability": 0.5, "contact": 0.0, "success": 100.0},
        2: {"dist": 1.0, "facing": 0.8, "stability": 0.5, "contact": 3.0, "success": 150.0},
        3: {"dist": 0.5, "facing": 0.5, "stability": 0.3, "contact": 5.0, "success": 200.0},
    }
    _WEIGHTS = stage_weights.get(stage, stage_weights[0])
    
    return 68, len(h0_act)


# =============================================================================
# Environment & PPO Logic
# =============================================================================

class EnvState(NamedTuple):
    data: mjx.Data
    step_count: jax.Array
    hug_hold: jax.Array
    rng: jax.Array

class Transition(NamedTuple):
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    done: jax.Array
    value: jax.Array
    log_prob: jax.Array

class PPONetwork(nn.Module):
    hidden_sizes: Tuple[int, ...]
    act_dim: int
    
    @nn.compact
    def __call__(self, x):
        a = x
        for sz in self.hidden_sizes:
            a = nn.tanh(nn.Dense(sz)(a))
        mean = nn.Dense(self.act_dim)(a)
        logstd = self.param("logstd", nn.initializers.zeros, (self.act_dim,))
        
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

def _reset_single(rng):
    rng, r1, r2, r3, r4, r5 = jax.random.split(rng, 6)
    data = mjx.make_data(_MJX_MODEL)
    qpos, qvel = data.qpos, jnp.zeros_like(data.qvel)
    
    dist = jax.random.uniform(r1, minval=0.75, maxval=1.25)
    
    s = _H0_QPOS_START
    h0_x = -dist + jax.random.uniform(r2, minval=-0.1, maxval=0.1)
    h0_y = jax.random.uniform(r2, minval=-0.1, maxval=0.1)
    yaw0 = jax.random.uniform(r3, minval=-0.2, maxval=0.2)
    qpos = qpos.at[s:s+3].set(jnp.array([h0_x, h0_y, 1.4]))
    qpos = qpos.at[s+3:s+7].set(jnp.array([jnp.cos(yaw0/2), 0, 0, jnp.sin(yaw0/2)]))
    
    s = _H1_QPOS_START
    h1_x = dist + jax.random.uniform(r4, minval=-0.1, maxval=0.1)
    h1_y = jax.random.uniform(r4, minval=-0.1, maxval=0.1)
    yaw1 = jnp.pi + jax.random.uniform(r5, minval=-0.2, maxval=0.2)
    qpos = qpos.at[s:s+3].set(jnp.array([h1_x, h1_y, 1.4]))
    qpos = qpos.at[s+3:s+7].set(jnp.array([jnp.cos(yaw1/2), 0, 0, jnp.sin(yaw1/2)]))
    
    data = data.replace(qpos=qpos, qvel=qvel)
    return mjx.forward(_MJX_MODEL, data)

def reset_env(rng, num_envs):
    reset_rngs = jax.random.split(rng, num_envs)
    data = jax.vmap(_reset_single)(reset_rngs)
    return EnvState(data, jnp.zeros(num_envs, int), jnp.zeros(num_envs, int), rng)

def _get_agent_obs(data, is_h0):
    if is_h0:
        qpos_s, qvel_s = _H0_QPOS_START, _H0_QVEL_START
        torso, partner = _H0_TORSO_IDX, _H1_TORSO_IDX
        p_qvel_s = _H1_QVEL_START
    else:
        qpos_s, qvel_s = _H1_QPOS_START, _H1_QVEL_START
        torso, partner = _H1_TORSO_IDX, _H0_TORSO_IDX
        p_qvel_s = _H0_QVEL_START
    
    jq = data.qpos[qpos_s+3:qpos_s+_NQ]
    jv = data.qvel[qvel_s+3:qvel_s+_NV]
    quat = data.qpos[qpos_s+3:qpos_s+7]
    avel = data.qvel[qvel_s+3:qvel_s+6]
    
    pos = data.xpos[torso]
    mat = data.xmat[torso].reshape(3, 3)
    p_pos = data.xpos[partner]
    p_mat = data.xmat[partner].reshape(3, 3)
    
    rel_pos = mat.T @ (p_pos - pos)
    rel_pel = mat.T @ (p_pos - jnp.array([0, 0, 0.2]) - pos)
    
    vel = data.qvel[qvel_s:qvel_s+3]
    p_vel = data.qvel[p_qvel_s:p_qvel_s+3]
    rel_vel = mat.T @ (p_vel - vel)
    
    face = -jnp.dot(mat[:, 0], p_mat[:, 0])
    up = jnp.dot(mat[:, 2], p_mat[:, 2])
    dist = jnp.linalg.norm(p_pos - pos)
    contact = (dist < 0.6).astype(jnp.float32)
    
    return jnp.concatenate([jq, jv, quat, avel, rel_pos, rel_pel, rel_vel, 
                          jnp.array([face, up]), jnp.full((3,), contact)])

def _get_obs(data):
    h0 = jax.vmap(lambda d: _get_agent_obs(d, True))(data)
    h1 = jax.vmap(lambda d: _get_agent_obs(d, False))(data)
    return {"h0": h0, "h1": h1}

def step_env(state, h0_act, h1_act):
    num_envs = h0_act.shape[0]
    ctrl = jnp.zeros((num_envs, _NU))
    ctrl = ctrl.at[:, _H0_ACT_IDX].set(jnp.clip(h0_act, -1, 1))
    ctrl = ctrl.at[:, _H1_ACT_IDX].set(jnp.clip(h1_act, -1, 1))
    
    data = state.data.replace(ctrl=ctrl)
    
    def phys_step(d, _): return mjx.step(_MJX_MODEL, d), None
    data, _ = jax.lax.scan(lambda d, _: jax.vmap(phys_step)(d, None), data, None, length=_FRAME_SKIP)
    
    h0_pos = data.xpos[:, _H0_TORSO_IDX]
    h1_pos = data.xpos[:, _H1_TORSO_IDX]
    dist = jnp.linalg.norm(h0_pos - h1_pos, axis=-1)
    
    hug_ok = (dist > 0.25) & (dist < 0.6)
    new_hold = jnp.where(hug_ok, state.hug_hold + 1, 0)
    success = new_hold >= 30
    
    h0_z = data.xpos[:, _H0_TORSO_IDX, 2]
    h1_z = data.xpos[:, _H1_TORSO_IDX, 2]
    fallen = (h0_z < 0.5) | (h1_z < 0.5)
    
    rew = _WEIGHTS["dist"] * jnp.exp(-3.0 * dist)
    rew += jnp.where(fallen, -100.0, 0.0)
    rew += jnp.where(success, _WEIGHTS["success"], 0.0)
    
    new_step = state.step_count + 1
    done = fallen | success | (new_step >= _HORIZON)
    
    rng, reset_rng = jax.random.split(state.rng)
    reset_rngs = jax.random.split(reset_rng, num_envs)
    
    def do_reset(d, flag, r):
        new = _reset_single(r)
        return jax.tree.map(lambda x, y: jnp.where(flag, y, x), d, new)
    
    data = jax.vmap(do_reset)(data, done, reset_rngs)
    new_step = jnp.where(done, 0, new_step)
    new_hold = jnp.where(done, 0, new_hold)
    
    return EnvState(data, new_step, new_hold, rng), _get_obs(data), rew, done, {"success": success}

def sample_action(rng, state, obs):
    mean, logstd, value = state.apply_fn(state.params, obs)
    std = jnp.exp(logstd)
    act = mean + std * jax.random.normal(rng, mean.shape)
    lp = -0.5 * jnp.sum(jnp.square((act - mean)/std) + 2*logstd + jnp.log(2*jnp.pi), -1)
    return act, lp, value.squeeze(-1)

def compute_gae(rewards, values, dones, last_val, gamma, lam):
    def scan(carry, t):
        gae, next_v = carry
        r, v, d = t
        delta = r + gamma * next_v * (1 - d) - v
        gae = delta + gamma * lam * (1 - d) * gae
        return (gae, v), gae
    _, adv = jax.lax.scan(scan, (jnp.zeros_like(last_val), last_val), (rewards[::-1], values[::-1], dones[::-1]))
    return adv[::-1], adv[::-1] + values

def train_update(runner_state, _):
    (env_state, obs, h0_state, h1_state, rng), config = runner_state
    
    # ROLLOUT
    def step_fn(carry, _):
        env_state, obs, rng = carry
        rng, r1, r2 = jax.random.split(rng, 3)
        h0_act, h0_lp, h0_val = sample_action(r1, h0_state, obs["h0"])
        h1_act, h1_lp, h1_val = sample_action(r2, h1_state, obs["h1"])
        env_state, next_obs, rew, done, info = step_env(env_state, h0_act, h1_act)
        return (env_state, next_obs, rng), (Transition(obs["h0"], h0_act, rew, done, h0_val, h0_lp),
                                          Transition(obs["h1"], h1_act, rew, done, h1_val, h1_lp), info)
    
    (env_state, obs, rng), (h0_traj, h1_traj, info) = jax.lax.scan(step_fn, (env_state, obs, rng), None, config.num_steps)
    
    # GAE
    rng, r1, r2 = jax.random.split(rng, 3)
    _, _, h0_last = sample_action(r1, h0_state, obs["h0"])
    _, _, h1_last = sample_action(r2, h1_state, obs["h1"])
    h0_adv, h0_ret = compute_gae(h0_traj.reward, h0_traj.value, h0_traj.done, h0_last, config.gamma, config.gae_lambda)
    h1_adv, h1_ret = compute_gae(h1_traj.reward, h1_traj.value, h1_traj.done, h1_last, config.gamma, config.gae_lambda)
    
    # UPDATE
    def update_ppo(state, batch, rng):
        obs, act, old_lp, adv, ret, _ = batch
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        def loss(params, x_obs, x_act, x_lp, x_adv, x_ret):
            mean, logstd, val = state.apply_fn(params, x_obs)
            std = jnp.exp(logstd)
            lp = -0.5 * jnp.sum(jnp.square((x_act - mean)/std) + 2*logstd + jnp.log(2*jnp.pi), -1)
            ratio = jnp.exp(lp - x_lp)
            pg = -jnp.minimum(x_adv * ratio, x_adv * jnp.clip(ratio, 0.8, 1.2)).mean()
            vl = 0.5 * jnp.square(val.squeeze(-1) - x_ret).mean()
            ent = 0.5 * jnp.sum(1 + jnp.log(2*jnp.pi) + 2*logstd).mean()
            return pg + config.vf_coef * vl - config.ent_coef * ent
        
        def epoch(state, rng):
            rng, p_rng = jax.random.split(rng)
            perm = jax.random.permutation(p_rng, obs.shape[0])
            def mb(state, i):
                idx = jax.lax.dynamic_slice(perm, (i,), (config.minibatch_size,))
                grads = jax.grad(loss)(state.params, obs[idx], act[idx], old_lp[idx], adv[idx], ret[idx])
                return state.apply_gradients(grads=grads), None
            state, _ = jax.lax.scan(mb, state, jnp.arange(0, obs.shape[0], config.minibatch_size))
            return state, None
        
        state, _ = jax.lax.scan(epoch, state, jax.random.split(rng, config.num_epochs))
        return state

    def flatten(traj, adv, ret):
        return tuple(x.reshape(-1, *x.shape[2:]) for x in (traj.obs, traj.action, traj.log_prob, adv, ret, traj.value))

    rng, r1, r2 = jax.random.split(rng, 3)
    h0_state = update_ppo(h0_state, flatten(h0_traj, h0_adv, h0_ret), r1)
    h1_state = update_ppo(h1_state, flatten(h1_traj, h1_adv, h1_ret), r2)
    
    return ((env_state, obs, h0_state, h1_state, rng), config), {"reward": h0_traj.reward.mean()}

def train(config: MJXConfig, checkpoint_dir: str, name: str):
    rng = jax.random.PRNGKey(config.seed)
    model = str(Path(__file__).parent.parent / "humanoid_hug" / "mjcf" / "humanoid_hug.xml")
    obs_dim, act_dim = init_env(model, config.frame_skip, config.horizon, config.stage)
    
    rng, r1, r2, r3 = jax.random.split(rng, 4)
    h0_state = create_train_state(r1, obs_dim, act_dim, config.hidden_sizes, config.learning_rate)
    h1_state = create_train_state(r2, obs_dim, act_dim, config.hidden_sizes, config.learning_rate)
    env_state = reset_env(r3, config.num_envs)
    obs = _get_obs(env_state.data)
    
    # JIT the *entire* training loop body
    train_step = jax.jit(train_update)
    runner = ((env_state, obs, h0_state, h1_state, rng), config)
    
    print("\nStarting training... (First update compiles JIT)")
    start = time.time()
    pbar = tqdm(range(config.num_updates))
    
    for i in pbar:
        runner, metrics = train_step(runner, None)
        # Block only for metrics
        jax.tree.map(lambda x: x.block_until_ready(), metrics)
        pbar.set_postfix({"rew": f"{metrics['reward']:.2f}"})
        
        if i % config.save_interval == 0:
            path = Path(checkpoint_dir) / f"{name}_{i}.npz"
            path.parent.mkdir(exist_ok=True, parents=True)
            jnp.savez(path, h0=runner[0][2].params, h1=runner[0][3].params)

def main():
    config = MJXConfig()
    train(config, "checkpoints_mjx", "humanoid_hug")

if __name__ == "__main__":
    main()
