#!/usr/bin/env python3
"""Evaluate a trained IPPO policy on the Humanoid Hug environment."""

import argparse
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from humanoid_hug import HumanoidHugEnv


def env_creator(config: Dict) -> ParallelPettingZooEnv:
    """Create and wrap the HumanoidHugEnv for RLlib."""
    env = HumanoidHugEnv(
        render_mode=config.get("render_mode", None),
        horizon=config.get("horizon", 1000),
        frame_skip=config.get("frame_skip", 5),
        hug_hold_target=config.get("hug_hold_target", 30),
        stage=config.get("stage", 0),
    )
    return ParallelPettingZooEnv(env)


def load_algorithm(checkpoint_path: str) -> Algorithm:
    """Load a trained algorithm from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint
        
    Returns:
        Loaded Algorithm instance
    """
    # Register the environment
    register_env("humanoid_hug", env_creator)
    
    # Load the algorithm
    algo = Algorithm.from_checkpoint(checkpoint_path)
    return algo


def get_actions(
    algo: Algorithm,
    observations: Dict[str, np.ndarray],
    share_policy: bool = False,
) -> Dict[str, np.ndarray]:
    """Get actions from the trained policy.
    
    Args:
        algo: Trained algorithm
        observations: Dictionary of observations per agent
        share_policy: Whether agents share the same policy
        
    Returns:
        Dictionary of actions per agent
    """
    actions = {}
    
    for agent_id, obs in observations.items():
        if share_policy:
            policy_id = "shared_policy"
        else:
            policy_id = f"{agent_id}_policy"
        
        action = algo.compute_single_action(
            obs,
            policy_id=policy_id,
            explore=False,  # Deterministic for evaluation
        )
        actions[agent_id] = action
    
    return actions


def run_evaluation_human(
    algo: Algorithm,
    env: HumanoidHugEnv,
    num_episodes: int = 5,
    share_policy: bool = False,
) -> Dict[str, float]:
    """Run evaluation with human rendering.
    
    Args:
        algo: Trained algorithm
        env: Environment instance
        num_episodes: Number of episodes to run
        share_policy: Whether agents share the same policy
        
    Returns:
        Dictionary of evaluation metrics
    """
    total_rewards = []
    episode_lengths = []
    successes = 0
    
    print("\nStarting evaluation with human rendering...")
    print("Close the viewer window to skip to next episode.\n")
    
    for ep in range(num_episodes):
        obs, infos = env.reset()
        episode_reward = 0
        step = 0
        
        print(f"Episode {ep + 1}/{num_episodes}")
        
        while env.agents:
            # Get actions from policy
            actions = get_actions(algo, obs, share_policy)
            
            # Step environment
            obs, rewards, terminations, truncations, infos = env.step(actions)
            
            episode_reward += sum(rewards.values()) / len(rewards) if rewards else 0
            step += 1
            
            # Render
            env.render()
            time.sleep(0.02)
            
            # Check for termination reason
            if any(terminations.values()) or any(truncations.values()):
                reason = infos.get("h0", {}).get("termination_reason", "unknown")
                if reason == "success":
                    successes += 1
                print(f"  Steps: {step}, Reward: {episode_reward:.2f}, Reason: {reason}")
        
        total_rewards.append(episode_reward)
        episode_lengths.append(step)
    
    metrics = {
        "mean_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "mean_length": np.mean(episode_lengths),
        "success_rate": successes / num_episodes,
    }
    
    return metrics


def run_evaluation_video(
    algo: Algorithm,
    env: HumanoidHugEnv,
    output_path: str,
    num_episodes: int = 3,
    share_policy: bool = False,
) -> Dict[str, float]:
    """Run evaluation and save to video.
    
    Args:
        algo: Trained algorithm
        env: Environment instance with rgb_array render mode
        output_path: Path to save video
        num_episodes: Number of episodes to record
        share_policy: Whether agents share the same policy
        
    Returns:
        Dictionary of evaluation metrics
    """
    if not HAS_IMAGEIO:
        print("Error: imageio not installed. Install with: pip install imageio imageio-ffmpeg")
        return {}
    
    total_rewards = []
    episode_lengths = []
    successes = 0
    frames = []
    
    print(f"\nRecording {num_episodes} episodes to {output_path}...")
    
    for ep in range(num_episodes):
        obs, infos = env.reset()
        episode_reward = 0
        step = 0
        
        while env.agents:
            # Get actions from policy
            actions = get_actions(algo, obs, share_policy)
            
            # Step environment
            obs, rewards, terminations, truncations, infos = env.step(actions)
            
            episode_reward += sum(rewards.values()) / len(rewards) if rewards else 0
            step += 1
            
            # Capture frame
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            
            # Check for termination reason
            if any(terminations.values()) or any(truncations.values()):
                reason = infos.get("h0", {}).get("termination_reason", "unknown")
                if reason == "success":
                    successes += 1
                print(f"Episode {ep + 1}: Steps={step}, Reward={episode_reward:.2f}, Reason={reason}")
        
        total_rewards.append(episode_reward)
        episode_lengths.append(step)
    
    # Save video
    if frames:
        print(f"Saving {len(frames)} frames...")
        imageio.mimsave(output_path, frames, fps=50)
        print(f"Video saved to {output_path}")
    
    metrics = {
        "mean_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "mean_length": np.mean(episode_lengths),
        "success_rate": successes / num_episodes,
    }
    
    return metrics


def run_evaluation_headless(
    algo: Algorithm,
    env: HumanoidHugEnv,
    num_episodes: int = 100,
    share_policy: bool = False,
) -> Dict[str, float]:
    """Run headless evaluation for metrics only.
    
    Args:
        algo: Trained algorithm
        env: Environment instance
        num_episodes: Number of episodes to evaluate
        share_policy: Whether agents share the same policy
        
    Returns:
        Dictionary of evaluation metrics
    """
    total_rewards = []
    episode_lengths = []
    successes = 0
    falls = 0
    hug_hold_steps = []
    
    print(f"\nRunning {num_episodes} evaluation episodes...")
    
    for ep in range(num_episodes):
        obs, infos = env.reset()
        episode_reward = 0
        step = 0
        max_hug_hold = 0
        
        while env.agents:
            # Get actions from policy
            actions = get_actions(algo, obs, share_policy)
            
            # Step environment
            obs, rewards, terminations, truncations, infos = env.step(actions)
            
            episode_reward += sum(rewards.values()) / len(rewards) if rewards else 0
            step += 1
            
            # Track hug hold steps
            current_hug_hold = infos.get("h0", {}).get("hug_hold_steps", 0)
            max_hug_hold = max(max_hug_hold, current_hug_hold)
            
            # Check for termination reason
            if any(terminations.values()) or any(truncations.values()):
                reason = infos.get("h0", {}).get("termination_reason", "unknown")
                if reason == "success":
                    successes += 1
                elif "fall" in str(reason):
                    falls += 1
        
        total_rewards.append(episode_reward)
        episode_lengths.append(step)
        hug_hold_steps.append(max_hug_hold)
        
        if (ep + 1) % 10 == 0:
            print(f"  Completed {ep + 1}/{num_episodes} episodes")
    
    metrics = {
        "mean_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "min_reward": np.min(total_rewards),
        "max_reward": np.max(total_rewards),
        "mean_length": np.mean(episode_lengths),
        "success_rate": successes / num_episodes,
        "fall_rate": falls / num_episodes,
        "mean_hug_hold_steps": np.mean(hug_hold_steps),
        "max_hug_hold_steps": np.max(hug_hold_steps),
    }
    
    return metrics


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate trained IPPO policy")
    
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--mode", choices=["human", "video", "headless"], default="headless",
                        help="Evaluation mode")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--output", type=str, default="eval_policy.mp4", help="Video output path")
    parser.add_argument("--stage", type=int, default=None, help="Override curriculum stage")
    parser.add_argument("--horizon", type=int, default=1000, help="Episode horizon")
    parser.add_argument("--share-policy", action="store_true", help="Use shared policy")
    
    args = parser.parse_args()
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    print("=" * 60)
    print("Humanoid Hug - Policy Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Mode: {args.mode}")
    print(f"Episodes: {args.num_episodes}")
    print("=" * 60)
    
    # Load algorithm
    print("\nLoading checkpoint...")
    algo = load_algorithm(args.checkpoint)
    
    # Get stage from checkpoint config if not overridden
    stage = args.stage
    if stage is None:
        stage = algo.config.env_config.get("stage", 0)
    
    # Create environment
    render_mode = None
    if args.mode == "human":
        render_mode = "human"
    elif args.mode == "video":
        render_mode = "rgb_array"
    
    env = HumanoidHugEnv(
        render_mode=render_mode,
        horizon=args.horizon,
        stage=stage,
    )
    
    # Run evaluation
    try:
        if args.mode == "human":
            metrics = run_evaluation_human(algo, env, args.num_episodes, args.share_policy)
        elif args.mode == "video":
            metrics = run_evaluation_video(algo, env, args.output, args.num_episodes, args.share_policy)
        else:
            metrics = run_evaluation_headless(algo, env, args.num_episodes, args.share_policy)
    finally:
        env.close()
        algo.stop()
        ray.shutdown()
    
    # Print metrics
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    main()
