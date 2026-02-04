#!/usr/bin/env python3
"""Curriculum training for Humanoid Hug - automatically progresses through stages."""

import argparse
import os
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env

from humanoid_hug import HumanoidHugEnv


def env_creator(config: Dict[str, Any]) -> ParallelPettingZooEnv:
    """Create and wrap the HumanoidHugEnv for RLlib."""
    env = HumanoidHugEnv(
        render_mode=config.get("render_mode", None),
        horizon=config.get("horizon", 1000),
        frame_skip=config.get("frame_skip", 5),
        hug_hold_target=config.get("hug_hold_target", 30),
        stage=config.get("stage", 0),
    )
    return ParallelPettingZooEnv(env)


def policy_mapping_fn(agent_id: str, episode, worker, **kwargs) -> str:
    """Map agent IDs to policy IDs."""
    return f"{agent_id}_policy"


# Stage progression thresholds
STAGE_THRESHOLDS = {
    0: {"reward": 25, "timesteps": 2_000_000},   # Approach
    1: {"reward": 45, "timesteps": 3_000_000},   # Reach
    2: {"reward": 70, "timesteps": 5_000_000},   # Contact
    3: {"reward": 150, "timesteps": 10_000_000}, # Full hug (final)
}


def create_config(
    stage: int,
    num_workers: int,
    num_gpus: float,
    train_batch_size: int,
    lr: float,
    horizon: int,
) -> PPOConfig:
    """Create PPO config for a given stage."""
    
    register_env("humanoid_hug", env_creator)
    
    env_config = {
        "horizon": horizon,
        "stage": stage,
        "frame_skip": 5,
        "hug_hold_target": 30,
    }
    
    # Create a temporary raw env to get individual agent spaces
    temp_raw_env = HumanoidHugEnv(
        render_mode=None,
        horizon=horizon,
        stage=stage,
    )
    obs_space = temp_raw_env.observation_space("h0")
    act_space = temp_raw_env.action_space("h0")
    temp_raw_env.close()
    
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            env="humanoid_hug",
            env_config=env_config,
        )
        .framework("torch")
        .resources(num_gpus=num_gpus)
        .env_runners(
            num_env_runners=num_workers,
            num_envs_per_env_runner=1,
            rollout_fragment_length="auto",
        )
        .training(
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01,
            vf_loss_coeff=0.5,
            grad_clip=0.5,
        )
        .multi_agent(
            policies={
                "h0_policy": PolicySpec(
                    observation_space=obs_space,
                    action_space=act_space,
                ),
                "h1_policy": PolicySpec(
                    observation_space=obs_space,
                    action_space=act_space,
                ),
            },
            policy_mapping_fn=policy_mapping_fn,
        )
    )
    
    # Set training hyperparameters directly (Ray 2.53+ compatibility)
    config.train_batch_size = train_batch_size
    config.sgd_minibatch_size = 256
    config.num_epochs = 10
    config.lr = lr
    config.gamma = 0.99
    config.model = {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "tanh",
        "free_log_std": True,
        "vf_share_layers": False,
    }
    
    return config


def should_advance_stage(
    current_stage: int,
    mean_reward: float,
    total_timesteps: int,
) -> bool:
    """Check if we should advance to the next curriculum stage."""
    if current_stage >= 3:
        return False
    
    threshold = STAGE_THRESHOLDS[current_stage]
    
    # Advance if reward threshold is met OR max timesteps for stage is reached
    reward_met = mean_reward >= threshold["reward"]
    timesteps_met = total_timesteps >= threshold["timesteps"]
    
    return reward_met or timesteps_met


def train_curriculum(
    num_workers: int = 4,
    num_gpus: float = 0,
    train_batch_size: int = 4000,
    lr: float = 3e-4,
    horizon: int = 1000,
    max_timesteps: int = 20_000_000,
    checkpoint_dir: str = "./checkpoints",
    start_stage: int = 0,
    resume_checkpoint: Optional[str] = None,
):
    """Run curriculum training through all stages."""
    
    checkpoint_path = Path(checkpoint_dir).resolve()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    current_stage = start_stage
    total_timesteps = 0
    algo = None
    
    # Load from checkpoint if resuming
    if resume_checkpoint:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        register_env("humanoid_hug", env_creator)
        algo = Algorithm.from_checkpoint(resume_checkpoint)
        current_stage = algo.config.env_config.get("stage", 0)
    
    print("=" * 60)
    print("Humanoid Hug - Curriculum Training")
    print("=" * 60)
    print(f"Starting stage: {current_stage}")
    print(f"Max timesteps: {max_timesteps:,}")
    print(f"Workers: {num_workers}, GPUs: {num_gpus}")
    print("=" * 60)
    
    while total_timesteps < max_timesteps and current_stage <= 3:
        print(f"\n{'='*60}")
        print(f"STAGE {current_stage}: {['Approach', 'Reach', 'Contact', 'Full Hug'][current_stage]}")
        print(f"{'='*60}")
        
        # Create or update config for current stage
        config = create_config(
            stage=current_stage,
            num_workers=num_workers,
            num_gpus=num_gpus,
            train_batch_size=train_batch_size,
            lr=lr,
            horizon=horizon,
        )
        
        # Create new algorithm or update existing
        if algo is None:
            algo = config.build()
        else:
            # Update environment config for new stage
            algo.config.env_config["stage"] = current_stage
            # Workers will pick up new stage on next reset
        
        stage_start_timesteps = total_timesteps
        iteration = 0
        recent_rewards = []
        
        while total_timesteps < max_timesteps:
            result = algo.train()
            iteration += 1
            
            # Get metrics
            timesteps = result.get("timesteps_total", result.get("num_env_steps_sampled_lifetime", 0))
            total_timesteps = timesteps
            mean_reward = result.get("env_runners/episode_reward_mean", 
                                    result.get("episode_reward_mean", 0))
            
            recent_rewards.append(mean_reward)
            if len(recent_rewards) > 20:
                recent_rewards.pop(0)
            smoothed_reward = np.mean(recent_rewards)
            
            # Progress logging
            if iteration % 10 == 0:
                print(f"  Iter {iteration}: timesteps={total_timesteps:,}, "
                      f"reward={mean_reward:.2f} (smooth: {smoothed_reward:.2f})")
            
            # Checkpoint
            if iteration % 50 == 0:
                ckpt_path = algo.save(str(checkpoint_path / f"stage{current_stage}"))
                print(f"  Checkpoint saved: {ckpt_path}")
            
            # Check stage advancement
            if should_advance_stage(current_stage, smoothed_reward, 
                                   total_timesteps - stage_start_timesteps):
                print(f"\n  Stage {current_stage} complete!")
                print(f"  Smoothed reward: {smoothed_reward:.2f}")
                print(f"  Stage timesteps: {total_timesteps - stage_start_timesteps:,}")
                
                # Save checkpoint before advancing
                ckpt_path = algo.save(str(checkpoint_path / f"stage{current_stage}_final"))
                print(f"  Final checkpoint: {ckpt_path}")
                
                current_stage += 1
                break
        
        if current_stage > 3:
            print("\n" + "=" * 60)
            print("TRAINING COMPLETE - All stages finished!")
            print("=" * 60)
    
    # Final save
    if algo:
        final_path = algo.save(str(checkpoint_path / "final"))
        print(f"\nFinal model saved: {final_path}")
        algo.stop()
    
    return final_path


def main():
    """Main entry point for curriculum training."""
    parser = argparse.ArgumentParser(description="Curriculum training for Humanoid Hug")
    
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--num-gpus", type=float, default=0, help="Number of GPUs")
    parser.add_argument("--train-batch-size", type=int, default=4000, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--horizon", type=int, default=1000, help="Episode horizon")
    parser.add_argument("--max-timesteps", type=int, default=20_000_000, help="Max timesteps")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Checkpoint dir")
    parser.add_argument("--start-stage", type=int, default=0, help="Starting stage (0-3)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    ray.init(ignore_reinit_error=True)
    
    try:
        train_curriculum(
            num_workers=args.num_workers,
            num_gpus=args.num_gpus,
            train_batch_size=args.train_batch_size,
            lr=args.lr,
            horizon=args.horizon,
            max_timesteps=args.max_timesteps,
            checkpoint_dir=args.checkpoint_dir,
            start_stage=args.start_stage,
            resume_checkpoint=args.resume,
        )
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
