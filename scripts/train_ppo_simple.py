#!/usr/bin/env python3
"""
Simple IPPO (Independent PPO) training for Humanoid Hug environment.
No Ray dependency - just PyTorch + PettingZoo.

Based on CleanRL's PPO implementation style.
"""

import argparse
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

from humanoid_hug import HumanoidHugEnv


@dataclass
class PPOConfig:
    """PPO hyperparameters."""
    # Environment
    horizon: int = 1000
    stage: int = 0
    
    # Training
    total_timesteps: int = 20_000_000
    num_envs: int = 8  # Parallel environments
    num_steps: int = 2048  # Steps per rollout per env
    num_epochs: int = 10  # PPO epochs per update
    minibatch_size: int = 512
    
    # PPO
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Network
    hidden_sizes: Tuple[int, ...] = (256, 256)
    
    # Logging
    log_interval: int = 1
    save_interval: int = 50
    
    # Misc
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps
    
    @property
    def num_minibatches(self) -> int:
        return self.batch_size // self.minibatch_size


class PPOAgent(nn.Module):
    """Actor-Critic network for PPO."""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Tuple[int, ...] = (256, 256)):
        super().__init__()
        
        # Actor (policy) network
        actor_layers = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes:
            actor_layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.Tanh(),
            ])
            prev_size = hidden_size
        self.actor_base = nn.Sequential(*actor_layers)
        self.actor_mean = nn.Linear(prev_size, act_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))
        
        # Critic (value) network
        critic_layers = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes:
            critic_layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.Tanh(),
            ])
            prev_size = hidden_size
        critic_layers.append(nn.Linear(prev_size, 1))
        self.critic = nn.Sequential(*critic_layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Smaller init for policy output
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)
    
    def get_action_and_value(
        self, 
        obs: torch.Tensor, 
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log prob, entropy, and value."""
        hidden = self.actor_base(obs)
        action_mean = self.actor_mean(hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        
        dist = Normal(action_mean, action_std)
        
        if action is None:
            if deterministic:
                action = action_mean
            else:
                action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(obs)
        
        return action, log_prob, entropy, value


def run_eval_episode(
    agents: Dict[str, "PPOAgent"],
    eval_env: HumanoidHugEnv,
    device: torch.device,
    output_path: Optional[str] = None,
    deterministic: bool = True,
) -> Tuple[float, int, str, bool]:
    """Run a single evaluation episode, optionally recording video.
    
    Args:
        agents: Dictionary of trained agents
        eval_env: Pre-created evaluation environment (reused to avoid init issues)
        device: Torch device
        output_path: Path to save video (None to skip recording)
        deterministic: Use deterministic actions
        
    Returns:
        Tuple of (episode_reward, episode_length, termination_reason, video_saved)
    """
    obs_dict, _ = eval_env.reset()
    
    frames = []
    episode_reward = 0.0
    episode_length = 0
    termination_reason = "unknown"
    can_render = eval_env.render_mode == "rgb_array"
    
    # Set agents to eval mode
    for agent in agents.values():
        agent.eval()
    
    while eval_env.agents:
        # Get observations as tensors
        obs = {
            agent_id: torch.tensor(obs_dict[agent_id], dtype=torch.float32, device=device).unsqueeze(0)
            for agent_id in ["h0", "h1"]
        }
        
        # Get actions
        with torch.no_grad():
            actions = {}
            for agent_id in ["h0", "h1"]:
                action, _, _, _ = agents[agent_id].get_action_and_value(
                    obs[agent_id], deterministic=deterministic
                )
                actions[agent_id] = action.cpu().numpy().squeeze(0)
        
        # Capture frame if rendering
        if can_render and output_path:
            try:
                frame = eval_env.render()
                if frame is not None:
                    frames.append(frame)
            except Exception:
                can_render = False
        
        # Step environment
        obs_dict, rewards, terminations, truncations, infos = eval_env.step(actions)
        
        episode_reward += (rewards.get("h0", 0) + rewards.get("h1", 0)) / 2
        episode_length += 1
        
        # Check done
        if any(terminations.values()) or any(truncations.values()):
            termination_reason = infos.get("h0", {}).get("termination_reason", "unknown")
            break
    
    # Set agents back to train mode
    for agent in agents.values():
        agent.train()
    
    # Save video if we have frames
    video_saved = False
    if frames and output_path and HAS_IMAGEIO:
        try:
            imageio.mimsave(output_path, frames, fps=50)
            video_saved = True
        except Exception:
            pass
    
    return episode_reward, episode_length, termination_reason, video_saved


class ParallelEnvWrapper:
    """Wrapper to run multiple HumanoidHugEnv instances in parallel."""
    
    def __init__(self, num_envs: int, horizon: int, stage: int):
        self.num_envs = num_envs
        self.envs = [
            HumanoidHugEnv(render_mode=None, horizon=horizon, stage=stage)
            for _ in range(num_envs)
        ]
        self.agents = ["h0", "h1"]
        
        # Get spaces from first env
        self.obs_dim = self.envs[0].observation_space("h0").shape[0]
        self.act_dim = self.envs[0].action_space("h0").shape[0]
    
    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Reset all environments."""
        obs_dict = {agent: [] for agent in self.agents}
        
        for i, env in enumerate(self.envs):
            env_seed = seed + i if seed is not None else None
            obs, _ = env.reset(seed=env_seed)
            for agent in self.agents:
                obs_dict[agent].append(obs[agent])
        
        return {agent: np.stack(obs_dict[agent]).astype(np.float32) for agent in self.agents}
    
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[
        Dict[str, np.ndarray],  # obs
        Dict[str, np.ndarray],  # rewards
        Dict[str, np.ndarray],  # dones
        List[Dict],  # infos
    ]:
        """Step all environments."""
        obs_dict = {agent: [] for agent in self.agents}
        reward_dict = {agent: [] for agent in self.agents}
        done_dict = {agent: [] for agent in self.agents}
        infos = []
        
        for i, env in enumerate(self.envs):
            # Get actions for this env
            env_actions = {agent: actions[agent][i] for agent in self.agents}
            
            # Check if env needs reset
            if not env.agents:
                obs, _ = env.reset()
                for agent in self.agents:
                    obs_dict[agent].append(obs[agent])
                    reward_dict[agent].append(0.0)
                    done_dict[agent].append(True)
                infos.append({})
                continue
            
            # Step
            obs, rewards, terminations, truncations, info = env.step(env_actions)
            
            # Handle episode end
            done = any(terminations.values()) or any(truncations.values())
            
            if done:
                # Store final info before reset
                final_info = info.get("h0", {})
                
                # Reset and get new obs
                obs, _ = env.reset()
                infos.append({"episode_end": True, **final_info})
            else:
                infos.append(info.get("h0", {}))
            
            for agent in self.agents:
                obs_dict[agent].append(obs[agent])
                reward_dict[agent].append(rewards.get(agent, 0.0))
                done_dict[agent].append(done)
        
        return (
            {agent: np.stack(obs_dict[agent]).astype(np.float32) for agent in self.agents},
            {agent: np.array(reward_dict[agent], dtype=np.float32) for agent in self.agents},
            {agent: np.array(done_dict[agent], dtype=bool) for agent in self.agents},
            infos,
        )
    
    def close(self):
        for env in self.envs:
            env.close()


class RolloutBuffer:
    """Buffer for storing rollout data."""
    
    def __init__(self, num_steps: int, num_envs: int, obs_dim: int, act_dim: int, device: str):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        
        # Storage
        self.obs = torch.zeros((num_steps, num_envs, obs_dim), device=device)
        self.actions = torch.zeros((num_steps, num_envs, act_dim), device=device)
        self.logprobs = torch.zeros((num_steps, num_envs), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)
        
        self.step = 0
    
    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        logprob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
    ):
        self.obs[self.step] = obs
        self.actions[self.step] = action
        self.logprobs[self.step] = logprob
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.values[self.step] = value.flatten()
        self.step += 1
    
    def reset(self):
        self.step = 0
    
    def compute_returns_and_advantages(
        self, 
        last_value: torch.Tensor, 
        last_done: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns."""
        advantages = torch.zeros_like(self.rewards)
        lastgaelam = 0
        
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                nextnonterminal = 1.0 - last_done.float()
                nextvalues = last_value.flatten()
            else:
                nextnonterminal = 1.0 - self.dones[t + 1]
                nextvalues = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        
        returns = advantages + self.values
        return returns, advantages


def train(config: PPOConfig, checkpoint_dir: str, experiment_name: str):
    """Main training loop."""
    
    # Seeding
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    
    # Setup
    device = torch.device(config.device)
    print(f"Using device: {device}")
    
    # Create parallel environments
    print(f"Creating {config.num_envs} parallel environments...")
    envs = ParallelEnvWrapper(config.num_envs, config.horizon, config.stage)
    
    # Create agents (one for each humanoid - IPPO)
    agents = {}
    optimizers = {}
    buffers = {}
    
    for agent_id in ["h0", "h1"]:
        agent = PPOAgent(envs.obs_dim, envs.act_dim, config.hidden_sizes).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)
        buffer = RolloutBuffer(config.num_steps, config.num_envs, envs.obs_dim, envs.act_dim, device)
        
        agents[agent_id] = agent
        optimizers[agent_id] = optimizer
        buffers[agent_id] = buffer
    
    # Logging
    run_name = f"{experiment_name}_{config.seed}_{int(time.time())}"
    log_dir = Path(checkpoint_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir / "tensorboard")
    
    print(f"Logging to: {log_dir}")
    print(f"TensorBoard: tensorboard --logdir {log_dir / 'tensorboard'}")
    
    # Create evaluation environment (reused throughout training to avoid init issues)
    eval_env = None
    can_record_video = False
    
    if HAS_IMAGEIO:
        try:
            eval_env = HumanoidHugEnv(render_mode="rgb_array", horizon=config.horizon, stage=config.stage)
            eval_env.reset()
            test_frame = eval_env.render()
            if test_frame is not None:
                can_record_video = True
                print(f"Videos will be saved to: {log_dir / 'videos'}")
            else:
                eval_env.close()
                eval_env = HumanoidHugEnv(render_mode=None, horizon=config.horizon, stage=config.stage)
                print("Video recording not available (render returned None)")
        except Exception as e:
            if eval_env:
                eval_env.close()
            eval_env = HumanoidHugEnv(render_mode=None, horizon=config.horizon, stage=config.stage)
            print(f"Video recording not available: {e}")
            print("Eval episodes will run without video.")
    else:
        eval_env = HumanoidHugEnv(render_mode=None, horizon=config.horizon, stage=config.stage)
        print("Warning: imageio not installed. Videos will not be recorded.")
        print("  Install with: pip install imageio imageio-ffmpeg")
    
    # Training state
    global_step = 0
    num_updates = config.total_timesteps // config.batch_size
    
    # Episode tracking
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    episode_count = 0
    
    # Initial reset
    obs_dict = envs.reset(seed=config.seed)
    obs = {agent_id: torch.tensor(obs_dict[agent_id], device=device) for agent_id in ["h0", "h1"]}
    dones = {agent_id: torch.zeros(config.num_envs, device=device) for agent_id in ["h0", "h1"]}
    
    # Track episode rewards per env
    current_ep_rewards = np.zeros(config.num_envs)
    current_ep_lengths = np.zeros(config.num_envs)
    
    print(f"\nStarting training for {config.total_timesteps:,} timesteps ({num_updates} updates)")
    print(f"Batch size: {config.batch_size}, Minibatch size: {config.minibatch_size}")
    print(f"Stage: {config.stage}, Horizon: {config.horizon}")
    print("=" * 60)
    
    # Main training loop with tqdm
    pbar = tqdm(range(1, num_updates + 1), desc="Training", unit="update")
    
    for update in pbar:
        update_start = time.time()
        
        # Annealing learning rate (optional, can be disabled)
        # frac = 1.0 - (update - 1.0) / num_updates
        # lr = frac * config.learning_rate
        # for opt in optimizers.values():
        #     for param_group in opt.param_groups:
        #         param_group["lr"] = lr
        
        # Collect rollouts
        for agent_id in ["h0", "h1"]:
            buffers[agent_id].reset()
        
        for step in range(config.num_steps):
            global_step += config.num_envs
            
            # Get actions from both agents
            actions_dict = {}
            with torch.no_grad():
                for agent_id in ["h0", "h1"]:
                    action, logprob, _, value = agents[agent_id].get_action_and_value(obs[agent_id])
                    actions_dict[agent_id] = action
                    
                    # Store in buffer
                    buffers[agent_id].add(
                        obs[agent_id],
                        action,
                        logprob,
                        torch.zeros(config.num_envs, device=device),  # Placeholder for reward
                        dones[agent_id],
                        value,
                    )
            
            # Step environment
            actions_np = {agent_id: actions_dict[agent_id].cpu().numpy() for agent_id in ["h0", "h1"]}
            next_obs_dict, rewards_dict, dones_dict, infos = envs.step(actions_np)
            
            # Update buffers with actual rewards
            for agent_id in ["h0", "h1"]:
                buffers[agent_id].rewards[step] = torch.tensor(rewards_dict[agent_id], device=device)
            
            # Track episodes
            for i, info in enumerate(infos):
                # Use mean reward across agents
                step_reward = (rewards_dict["h0"][i] + rewards_dict["h1"][i]) / 2
                current_ep_rewards[i] += step_reward
                current_ep_lengths[i] += 1
                
                if info.get("episode_end", False):
                    episode_rewards.append(current_ep_rewards[i])
                    episode_lengths.append(current_ep_lengths[i])
                    episode_count += 1
                    current_ep_rewards[i] = 0
                    current_ep_lengths[i] = 0
            
            # Update state
            obs = {agent_id: torch.tensor(next_obs_dict[agent_id], device=device) for agent_id in ["h0", "h1"]}
            dones = {agent_id: torch.tensor(dones_dict[agent_id], dtype=torch.float32, device=device) for agent_id in ["h0", "h1"]}
        
        # Compute advantages for both agents
        returns_dict = {}
        advantages_dict = {}
        
        with torch.no_grad():
            for agent_id in ["h0", "h1"]:
                last_value = agents[agent_id].get_value(obs[agent_id])
                returns, advantages = buffers[agent_id].compute_returns_and_advantages(
                    last_value, dones[agent_id], config.gamma, config.gae_lambda
                )
                returns_dict[agent_id] = returns
                advantages_dict[agent_id] = advantages
        
        # PPO update for both agents
        total_pg_loss = 0
        total_v_loss = 0
        total_entropy = 0
        total_clipfrac = 0
        
        for agent_id in ["h0", "h1"]:
            buffer = buffers[agent_id]
            returns = returns_dict[agent_id]
            advantages = advantages_dict[agent_id]
            
            # Flatten batch
            b_obs = buffer.obs.reshape(-1, envs.obs_dim)
            b_actions = buffer.actions.reshape(-1, envs.act_dim)
            b_logprobs = buffer.logprobs.reshape(-1)
            b_returns = returns.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_values = buffer.values.reshape(-1)
            
            # Normalize advantages
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
            
            # PPO epochs
            b_inds = np.arange(config.batch_size)
            
            for epoch in range(config.num_epochs):
                np.random.shuffle(b_inds)
                
                for start in range(0, config.batch_size, config.minibatch_size):
                    end = start + config.minibatch_size
                    mb_inds = b_inds[start:end]
                    
                    _, newlogprob, entropy, newvalue = agents[agent_id].get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                    
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    
                    # Clipped surrogate loss
                    mb_advantages = b_advantages[mb_inds]
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    
                    # Value loss
                    newvalue = newvalue.view(-1)
                    if config.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds], -config.clip_coef, config.clip_coef
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                    
                    # Entropy loss
                    entropy_loss = entropy.mean()
                    
                    # Total loss
                    loss = pg_loss - config.ent_coef * entropy_loss + config.vf_coef * v_loss
                    
                    # Update
                    optimizers[agent_id].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agents[agent_id].parameters(), config.max_grad_norm)
                    optimizers[agent_id].step()
                    
                    # Track metrics
                    with torch.no_grad():
                        clipfrac = ((ratio - 1.0).abs() > config.clip_coef).float().mean().item()
                        total_clipfrac += clipfrac
            
            total_pg_loss += pg_loss.item()
            total_v_loss += v_loss.item()
            total_entropy += entropy_loss.item()
        
        # Average metrics across agents
        num_agent_updates = 2 * config.num_epochs * config.num_minibatches
        avg_clipfrac = total_clipfrac / num_agent_updates
        
        update_time = time.time() - update_start
        fps = config.batch_size / update_time
        
        # Logging
        if update % config.log_interval == 0 and len(episode_rewards) > 0:
            mean_reward = np.mean(episode_rewards)
            mean_length = np.mean(episode_lengths)
            
            writer.add_scalar("charts/mean_reward", mean_reward, global_step)
            writer.add_scalar("charts/mean_episode_length", mean_length, global_step)
            writer.add_scalar("charts/episodes", episode_count, global_step)
            writer.add_scalar("losses/policy_loss", total_pg_loss / 2, global_step)
            writer.add_scalar("losses/value_loss", total_v_loss / 2, global_step)
            writer.add_scalar("losses/entropy", total_entropy / 2, global_step)
            writer.add_scalar("losses/clipfrac", avg_clipfrac, global_step)
            writer.add_scalar("charts/fps", fps, global_step)
            
            # Update progress bar
            pbar.set_postfix({
                "reward": f"{mean_reward:.1f}",
                "ep_len": f"{mean_length:.0f}",
                "fps": f"{fps:.0f}",
            })
        
        # Save checkpoint and video
        if update % config.save_interval == 0:
            checkpoint_path = log_dir / f"checkpoint_{update}.pt"
            torch.save({
                "update": update,
                "global_step": global_step,
                "h0_state_dict": agents["h0"].state_dict(),
                "h1_state_dict": agents["h1"].state_dict(),
                "h0_optimizer": optimizers["h0"].state_dict(),
                "h1_optimizer": optimizers["h1"].state_dict(),
                "config": config,
            }, checkpoint_path)
            tqdm.write(f"Saved checkpoint: {checkpoint_path}")
            
            # Run evaluation episode (with video if rendering available)
            video_dir = log_dir / "videos"
            video_dir.mkdir(exist_ok=True)
            video_path = video_dir / f"episode_{update}.mp4" if can_record_video else None
            
            vid_reward, vid_length, vid_reason, video_saved = run_eval_episode(
                agents, eval_env, device, str(video_path) if video_path else None, deterministic=True
            )
            
            if video_saved:
                tqdm.write(f"Saved video: {video_path} (reward={vid_reward:.1f}, len={vid_length}, {vid_reason})")
            else:
                tqdm.write(f"Eval episode: reward={vid_reward:.1f}, len={vid_length}, {vid_reason}")
            
            # Log eval metrics
            writer.add_scalar("eval/episode_reward", vid_reward, global_step)
            writer.add_scalar("eval/episode_length", vid_length, global_step)
    
    # Final save
    final_path = log_dir / "final_model.pt"
    torch.save({
        "update": num_updates,
        "global_step": global_step,
        "h0_state_dict": agents["h0"].state_dict(),
        "h1_state_dict": agents["h1"].state_dict(),
        "config": config,
    }, final_path)
    print(f"\nTraining complete! Final model saved to: {final_path}")
    
    # Cleanup
    writer.close()
    envs.close()
    if eval_env:
        eval_env.close()


def main():
    parser = argparse.ArgumentParser(description="Simple PPO training for Humanoid Hug")
    
    # Environment
    parser.add_argument("--stage", type=int, default=0, help="Curriculum stage (0-3)")
    parser.add_argument("--horizon", type=int, default=1000, help="Episode horizon")
    
    # Training
    parser.add_argument("--total-timesteps", type=int, default=20_000_000, help="Total timesteps")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--num-steps", type=int, default=2048, help="Steps per rollout")
    parser.add_argument("--num-epochs", type=int, default=10, help="PPO epochs per update")
    parser.add_argument("--minibatch-size", type=int, default=512, help="Minibatch size")
    
    # PPO hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="PPO clip coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")
    
    # Network
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256], help="Hidden layer sizes")
    
    # Logging
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--experiment-name", type=str, default="humanoid_hug_ppo", help="Experiment name")
    parser.add_argument("--log-interval", type=int, default=1, help="Log every N updates")
    parser.add_argument("--save-interval", type=int, default=50, help="Save every N updates")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    config = PPOConfig(
        horizon=args.horizon,
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
        device=args.device,
    )
    
    print("=" * 60)
    print("Humanoid Hug - Simple PPO Training")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Stage: {config.stage}")
    print(f"Total timesteps: {config.total_timesteps:,}")
    print(f"Parallel envs: {config.num_envs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Network: {config.hidden_sizes}")
    print("=" * 60)
    
    train(config, args.checkpoint_dir, args.experiment_name)


if __name__ == "__main__":
    main()
