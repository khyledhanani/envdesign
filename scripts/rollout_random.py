#!/usr/bin/env python3
"""Run random rollouts in the humanoid hug environment."""

import argparse
import numpy as np

from humanoid_hug import HumanoidHugEnv


def run_random_rollout(env: HumanoidHugEnv, seed: int) -> dict:
    """Run a single episode with random actions.

    Args:
        env: The environment instance
        seed: Random seed for this episode

    Returns:
        Dictionary with episode statistics
    """
    obs, infos = env.reset(seed=seed)

    total_reward = 0.0
    step_count = 0
    max_hug_hold = 0
    termination_reason = "horizon"

    while env.agents:
        # Random actions for each agent
        actions = {}
        for agent in env.agents:
            actions[agent] = env.action_space(agent).sample()

        obs, rewards, terminations, truncations, infos = env.step(actions)

        # Accumulate reward (same for both agents)
        if "h0" in rewards:
            total_reward += rewards["h0"]

        step_count += 1

        # Track max hug hold steps
        if "h0" in infos and "hug_hold_steps" in infos["h0"]:
            max_hug_hold = max(max_hug_hold, infos["h0"]["hug_hold_steps"])

        # Get termination reason
        if "h0" in infos and "termination_reason" in infos["h0"]:
            if infos["h0"]["termination_reason"] is not None:
                termination_reason = infos["h0"]["termination_reason"]

    return {
        "total_reward": total_reward,
        "steps": step_count,
        "max_hug_hold_steps": max_hug_hold,
        "termination_reason": termination_reason,
    }


def main():
    """Run multiple random rollouts and print statistics."""
    parser = argparse.ArgumentParser(description="Run random rollouts in humanoid hug env")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    parser.add_argument("--horizon", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--stage", type=int, default=0, help="Curriculum stage (0-3)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    args = parser.parse_args()

    print("=" * 60)
    print("Humanoid Hug Environment - Random Rollout")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"Horizon: {args.horizon}")
    print(f"Stage: {args.stage}")
    print(f"Seed: {args.seed}")
    print("=" * 60)

    env = HumanoidHugEnv(
        render_mode=None,
        horizon=args.horizon,
        stage=args.stage,
    )

    all_results = []

    for ep in range(args.episodes):
        episode_seed = args.seed + ep
        print(f"\nEpisode {ep + 1}/{args.episodes} (seed={episode_seed})")
        print("-" * 40)

        result = run_random_rollout(env, episode_seed)
        all_results.append(result)

        print(f"  Total reward:       {result['total_reward']:.2f}")
        print(f"  Steps:              {result['steps']}")
        print(f"  Max hug hold steps: {result['max_hug_hold_steps']}")
        print(f"  Termination:        {result['termination_reason']}")

    env.close()

    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    rewards = [r["total_reward"] for r in all_results]
    steps = [r["steps"] for r in all_results]
    hug_holds = [r["max_hug_hold_steps"] for r in all_results]

    print(f"Reward:    mean={np.mean(rewards):.2f}, std={np.std(rewards):.2f}, "
          f"min={np.min(rewards):.2f}, max={np.max(rewards):.2f}")
    print(f"Steps:     mean={np.mean(steps):.1f}, std={np.std(steps):.1f}")
    print(f"Hug hold:  mean={np.mean(hug_holds):.1f}, max={np.max(hug_holds)}")

    termination_counts = {}
    for r in all_results:
        reason = r["termination_reason"]
        termination_counts[reason] = termination_counts.get(reason, 0) + 1

    print("\nTermination reasons:")
    for reason, count in sorted(termination_counts.items()):
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
