#!/usr/bin/env python3
"""
Evaluate trained PPO model from train_ppo_simple.py
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

from humanoid_hug import HumanoidHugEnv
from train_ppo_simple import PPOAgent, PPOConfig


def load_model(checkpoint_path: str, device: str = "cpu"):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config
    config = checkpoint.get("config", PPOConfig())
    
    # Create environment to get dimensions
    env = HumanoidHugEnv(render_mode=None, horizon=100, stage=0)
    obs_dim = env.observation_space("h0").shape[0]
    act_dim = env.action_space("h0").shape[0]
    env.close()
    
    # Create agents
    hidden_sizes = config.hidden_sizes if hasattr(config, 'hidden_sizes') else (256, 256)
    agents = {
        "h0": PPOAgent(obs_dim, act_dim, hidden_sizes).to(device),
        "h1": PPOAgent(obs_dim, act_dim, hidden_sizes).to(device),
    }
    
    # Load weights
    agents["h0"].load_state_dict(checkpoint["h0_state_dict"])
    agents["h1"].load_state_dict(checkpoint["h1_state_dict"])
    
    # Set to eval mode
    agents["h0"].eval()
    agents["h1"].eval()
    
    return agents, config


def run_episode(
    env: HumanoidHugEnv, 
    agents: dict, 
    device: str,
    deterministic: bool = True,
    render: bool = False,
    record_frames: bool = False,
):
    """Run a single episode."""
    obs_dict, _ = env.reset()
    
    episode_reward = 0
    episode_length = 0
    frames = []
    
    while env.agents:
        # Get observations
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
        
        # Step
        obs_dict, rewards, terminations, truncations, infos = env.step(actions)
        
        episode_reward += (rewards.get("h0", 0) + rewards.get("h1", 0)) / 2
        episode_length += 1
        
        # Render
        if render:
            env.render()
            time.sleep(0.02)
        
        if record_frames:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        
        # Check done
        if any(terminations.values()) or any(truncations.values()):
            break
    
    termination_reason = infos.get("h0", {}).get("termination_reason", "unknown")
    
    return episode_reward, episode_length, termination_reason, frames


def evaluate_human(agents: dict, device: str, num_episodes: int, stage: int, horizon: int):
    """Run evaluation with human rendering."""
    env = HumanoidHugEnv(render_mode="human", horizon=horizon, stage=stage)
    
    print(f"\nRunning {num_episodes} episodes with human rendering...")
    print("Close the viewer window to skip to next episode.\n")
    
    rewards = []
    lengths = []
    successes = 0
    
    for ep in range(num_episodes):
        reward, length, reason, _ = run_episode(env, agents, device, render=True)
        rewards.append(reward)
        lengths.append(length)
        if reason == "success":
            successes += 1
        print(f"Episode {ep + 1}: Reward={reward:.2f}, Length={length}, Reason={reason}")
    
    env.close()
    
    print(f"\nResults over {num_episodes} episodes:")
    print(f"  Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Mean length: {np.mean(lengths):.1f}")
    print(f"  Success rate: {successes}/{num_episodes} ({100*successes/num_episodes:.1f}%)")


def evaluate_video(agents: dict, device: str, num_episodes: int, stage: int, horizon: int, output_path: str):
    """Run evaluation and save to video."""
    if not HAS_IMAGEIO:
        print("Error: imageio not installed. Install with: pip install imageio imageio-ffmpeg")
        return
    
    env = HumanoidHugEnv(render_mode="rgb_array", horizon=horizon, stage=stage)
    
    print(f"\nRecording {num_episodes} episodes to {output_path}...")
    
    all_frames = []
    rewards = []
    
    for ep in range(num_episodes):
        reward, length, reason, frames = run_episode(env, agents, device, record_frames=True)
        rewards.append(reward)
        all_frames.extend(frames)
        print(f"Episode {ep + 1}: Reward={reward:.2f}, Length={length}, Reason={reason}")
    
    env.close()
    
    if all_frames:
        print(f"Saving {len(all_frames)} frames...")
        imageio.mimsave(output_path, all_frames, fps=50)
        print(f"Video saved to {output_path}")
    
    print(f"\nMean reward: {np.mean(rewards):.2f}")


def evaluate_headless(agents: dict, device: str, num_episodes: int, stage: int, horizon: int):
    """Run headless evaluation for metrics."""
    env = HumanoidHugEnv(render_mode=None, horizon=horizon, stage=stage)
    
    print(f"\nRunning {num_episodes} headless evaluation episodes...")
    
    rewards = []
    lengths = []
    successes = 0
    falls = 0
    
    for ep in range(num_episodes):
        reward, length, reason, _ = run_episode(env, agents, device)
        rewards.append(reward)
        lengths.append(length)
        
        if reason == "success":
            successes += 1
        elif "fall" in str(reason):
            falls += 1
        
        if (ep + 1) % 10 == 0:
            print(f"  Completed {ep + 1}/{num_episodes} episodes")
    
    env.close()
    
    print(f"\n{'='*40}")
    print("Evaluation Results")
    print(f"{'='*40}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Min/Max reward: {np.min(rewards):.2f} / {np.max(rewards):.2f}")
    print(f"  Mean length: {np.mean(lengths):.1f}")
    print(f"  Success rate: {100*successes/num_episodes:.1f}%")
    print(f"  Fall rate: {100*falls/num_episodes:.1f}%")
    print(f"{'='*40}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained PPO model")
    
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--mode", choices=["human", "video", "headless"], default="headless",
                        help="Evaluation mode")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--output", type=str, default="eval_video.mp4", help="Video output path")
    parser.add_argument("--stage", type=int, default=None, help="Override curriculum stage")
    parser.add_argument("--horizon", type=int, default=1000, help="Episode horizon")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Humanoid Hug - PPO Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Mode: {args.mode}")
    print(f"Device: {args.device}")
    
    # Load model
    print("\nLoading model...")
    agents, config = load_model(args.checkpoint, args.device)
    
    # Get stage
    stage = args.stage if args.stage is not None else getattr(config, 'stage', 0)
    print(f"Stage: {stage}")
    
    # Run evaluation
    if args.mode == "human":
        evaluate_human(agents, args.device, args.num_episodes, stage, args.horizon)
    elif args.mode == "video":
        evaluate_video(agents, args.device, args.num_episodes, stage, args.horizon, args.output)
    else:
        evaluate_headless(agents, args.device, args.num_episodes, stage, args.horizon)


if __name__ == "__main__":
    main()
