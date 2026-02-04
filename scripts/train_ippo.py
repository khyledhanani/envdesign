#!/usr/bin/env python3
"""Train IPPO (Independent PPO) on the Humanoid Hug environment using RLlib."""

import argparse
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env

from humanoid_hug import HumanoidHugEnv


def env_creator(config: Dict[str, Any]) -> ParallelPettingZooEnv:
    """Create and wrap the HumanoidHugEnv for RLlib.
    
    Args:
        config: Environment configuration dict
        
    Returns:
        Wrapped PettingZoo environment
    """
    env = HumanoidHugEnv(
        render_mode=config.get("render_mode", None),
        horizon=config.get("horizon", 1000),
        frame_skip=config.get("frame_skip", 5),
        hug_hold_target=config.get("hug_hold_target", 30),
        stage=config.get("stage", 0),
    )
    return ParallelPettingZooEnv(env)


def get_policy_config(env: ParallelPettingZooEnv) -> Dict[str, PolicySpec]:
    """Get multi-agent policy configuration.
    
    For IPPO, each agent gets its own independent policy.
    
    Args:
        env: The wrapped environment
        
    Returns:
        Dictionary mapping policy IDs to PolicySpec
    """
    # Get spaces from the underlying env
    obs_space = env.observation_space
    act_space = env.action_space
    
    return {
        "h0_policy": PolicySpec(
            observation_space=obs_space,
            action_space=act_space,
        ),
        "h1_policy": PolicySpec(
            observation_space=obs_space,
            action_space=act_space,
        ),
    }


def policy_mapping_fn(agent_id: str, episode, worker, **kwargs) -> str:
    """Map agent IDs to policy IDs.
    
    For IPPO, each agent uses its own independent policy.
    For parameter sharing, both agents could use the same policy.
    
    Args:
        agent_id: The agent identifier ("h0" or "h1")
        
    Returns:
        Policy ID string
    """
    return f"{agent_id}_policy"


def policy_mapping_fn_shared(agent_id: str, episode, worker, **kwargs) -> str:
    """Map all agents to a shared policy (for parameter sharing variant).
    
    Args:
        agent_id: The agent identifier
        
    Returns:
        Shared policy ID
    """
    return "shared_policy"


def create_ippo_config(
    env_config: Dict[str, Any],
    num_workers: int = 4,
    num_gpus: float = 0,
    train_batch_size: int = 4000,
    minibatch_size: int = 256,
    num_epochs: int = 10,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_param: float = 0.2,
    entropy_coeff: float = 0.01,
    vf_loss_coeff: float = 0.5,
    grad_clip: float = 0.5,
    share_policy: bool = False,
    fcnet_hiddens: list = None,
) -> PPOConfig:
    """Create PPO configuration for IPPO training.
    
    Args:
        env_config: Environment configuration
        num_workers: Number of parallel rollout workers
        num_gpus: Number of GPUs to use (can be fractional)
        train_batch_size: Total batch size for training
        minibatch_size: Minibatch size for SGD
        num_epochs: Number of SGD epochs per train batch
        lr: Learning rate
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_param: PPO clip parameter
        entropy_coeff: Entropy coefficient for exploration
        vf_loss_coeff: Value function loss coefficient
        grad_clip: Gradient clipping value
        share_policy: Whether to share policy between agents
        fcnet_hiddens: Hidden layer sizes for the policy network
        
    Returns:
        Configured PPOConfig
    """
    if fcnet_hiddens is None:
        fcnet_hiddens = [256, 256]
    
    # Register the environment
    register_env("humanoid_hug", env_creator)
    
    # Create a temporary raw env to get individual agent spaces
    # (The wrapped env has Dict spaces, but policies need individual Box spaces)
    temp_raw_env = HumanoidHugEnv(
        render_mode=None,
        horizon=env_config.get("horizon", 1000),
        stage=env_config.get("stage", 0),
    )
    obs_space = temp_raw_env.observation_space("h0")
    act_space = temp_raw_env.action_space("h0")
    temp_raw_env.close()
    
    # Build config - use old API stack for stability with multi-agent PettingZoo envs
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
        .resources(
            num_gpus=num_gpus,
        )
        .env_runners(
            num_env_runners=num_workers,
            num_envs_per_env_runner=1,
            rollout_fragment_length="auto",
        )
        .training(
            lambda_=gae_lambda,
            clip_param=clip_param,
            entropy_coeff=entropy_coeff,
            vf_loss_coeff=vf_loss_coeff,
            grad_clip=grad_clip,
        )
    )
    
    # Set training hyperparameters directly (Ray 2.53+ compatibility)
    config.train_batch_size = train_batch_size
    config.sgd_minibatch_size = minibatch_size
    config.num_epochs = num_epochs
    config.lr = lr
    config.gamma = gamma
    config.model = {
        "fcnet_hiddens": fcnet_hiddens,
        "fcnet_activation": "tanh",
        "free_log_std": True,
        "vf_share_layers": False,
    }
    
    # Multi-agent configuration
    if share_policy:
        # Parameter sharing: both agents use the same policy
        config = config.multi_agent(
            policies={
                "shared_policy": PolicySpec(
                    observation_space=obs_space,
                    action_space=act_space,
                ),
            },
            policy_mapping_fn=policy_mapping_fn_shared,
        )
    else:
        # IPPO: independent policies for each agent
        config = config.multi_agent(
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
    
    return config


def train(
    config: PPOConfig,
    stop_config: Dict[str, Any],
    checkpoint_dir: str,
    experiment_name: str = "humanoid_hug_ippo",
    resume: bool = False,
    verbose: int = 1,
) -> tune.ResultGrid:
    """Run IPPO training.
    
    Args:
        config: PPO configuration
        stop_config: Stopping criteria
        checkpoint_dir: Directory to save checkpoints
        experiment_name: Name for the experiment
        resume: Whether to resume from checkpoint
        verbose: Verbosity level
        
    Returns:
        Tune results
    """
    results = tune.run(
        "PPO",
        name=experiment_name,
        config=config.to_dict(),
        stop=stop_config,
        storage_path=checkpoint_dir,
        checkpoint_freq=50,
        checkpoint_at_end=True,
        verbose=verbose,
        resume=resume if resume else None,
    )
    
    return results


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train IPPO on Humanoid Hug")
    
    # Environment args
    parser.add_argument("--stage", type=int, default=0, help="Curriculum stage (0-3)")
    parser.add_argument("--horizon", type=int, default=1000, help="Episode horizon")
    
    # Training args
    parser.add_argument("--num-workers", type=int, default=4, help="Number of rollout workers")
    parser.add_argument("--num-gpus", type=float, default=0, help="Number of GPUs (can be fractional)")
    parser.add_argument("--train-batch-size", type=int, default=4000, help="Training batch size")
    parser.add_argument("--minibatch-size", type=int, default=256, help="SGD minibatch size")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of SGD epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--clip-param", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--entropy-coeff", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--grad-clip", type=float, default=0.5, help="Gradient clipping")
    parser.add_argument("--share-policy", action="store_true", help="Share policy between agents")
    parser.add_argument("--fcnet-hiddens", type=int, nargs="+", default=[256, 256], help="Network hidden sizes")
    
    # Stopping criteria
    parser.add_argument("--stop-iters", type=int, default=1000, help="Max training iterations")
    parser.add_argument("--stop-timesteps", type=int, default=10_000_000, help="Max timesteps")
    parser.add_argument("--stop-reward", type=float, default=200, help="Target mean reward")
    
    # Output args
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--experiment-name", type=str, default="humanoid_hug_ippo", help="Experiment name")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    print("=" * 60)
    print("Humanoid Hug - IPPO Training")
    print("=" * 60)
    print(f"Stage: {args.stage}")
    print(f"Horizon: {args.horizon}")
    print(f"Workers: {args.num_workers}")
    print(f"GPUs: {args.num_gpus}")
    print(f"Batch size: {args.train_batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Policy sharing: {args.share_policy}")
    print(f"Network: {args.fcnet_hiddens}")
    print("=" * 60)
    
    # Environment config
    env_config = {
        "horizon": args.horizon,
        "stage": args.stage,
        "frame_skip": 5,
        "hug_hold_target": 30,
    }
    
    # Create IPPO config
    config = create_ippo_config(
        env_config=env_config,
        num_workers=args.num_workers,
        num_gpus=args.num_gpus,
        train_batch_size=args.train_batch_size,
        minibatch_size=args.minibatch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        gamma=args.gamma,
        clip_param=args.clip_param,
        entropy_coeff=args.entropy_coeff,
        grad_clip=args.grad_clip,
        share_policy=args.share_policy,
        fcnet_hiddens=args.fcnet_hiddens,
    )
    
    # Stopping criteria
    stop_config = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "env_runners/episode_reward_mean": args.stop_reward,
    }
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Train
    print("\nStarting training...")
    results = train(
        config=config,
        stop_config=stop_config,
        checkpoint_dir=str(checkpoint_dir),
        experiment_name=args.experiment_name,
        resume=args.resume,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    try:
        best_result = results.best_result
        if best_result:
            best_reward = best_result.get("env_runners/episode_reward_mean", 
                          best_result.get("episode_reward_mean", "N/A"))
            print(f"Best reward: {best_reward}")
            best_checkpoint = results.best_checkpoint
            if best_checkpoint:
                print(f"Best checkpoint: {best_checkpoint}")
    except Exception as e:
        print(f"Could not retrieve best result: {e}")
        print(f"Check {checkpoint_dir} for saved checkpoints")
    
    ray.shutdown()


if __name__ == "__main__":
    main()
