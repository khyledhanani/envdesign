#!/usr/bin/env python3
"""Render a demo of the humanoid hug environment."""

import argparse
import time
import numpy as np

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

from humanoid_hug import HumanoidHugEnv


def run_demo_human(env: HumanoidHugEnv, max_steps: int = 500) -> None:
    """Run a demo with human rendering (viewer window).

    Args:
        env: Environment with render_mode='human'
        max_steps: Maximum steps to run
    """
    obs, infos = env.reset(seed=42)
    print("Starting human-rendered demo...")
    print("Close the viewer window to exit.")

    step = 0
    episode_num = 1
    episode_step = 0
    
    while step < max_steps:
        # Reset if no agents (episode terminated)
        if not env.agents:
            print(f"\nEpisode {episode_num} ended after {episode_step} steps")
            episode_num += 1
            episode_step = 0
            obs, infos = env.reset()
            print(f"Starting episode {episode_num}...")
        
        # Random actions
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)

        env.render()
        time.sleep(0.02)  # ~50 FPS
        step += 1
        episode_step += 1

        if step % 100 == 0:
            reward_val = list(rewards.values())[0] if rewards else 0.0
            print(f"Step {step}, Episode {episode_num}, Reward: {reward_val:.2f}")

    print(f"\nDemo ended after {step} total steps across {episode_num} episodes")


def run_demo_video(env: HumanoidHugEnv, output_path: str, max_steps: int = 500) -> None:
    """Run a demo and save to video file.

    Args:
        env: Environment with render_mode='rgb_array'
        output_path: Path to save video
        max_steps: Maximum steps to run
    """
    if not HAS_IMAGEIO:
        print("Error: imageio not installed. Install with: pip install imageio imageio-ffmpeg")
        return

    obs, infos = env.reset(seed=42)
    print(f"Recording video to {output_path}...")

    frames = []
    step = 0
    episode_num = 1

    while step < max_steps:
        # Reset if no agents (episode terminated)
        if not env.agents:
            print(f"Episode {episode_num} ended, resetting...")
            episode_num += 1
            obs, infos = env.reset()
        
        # Random actions
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)

        # Capture frame
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        step += 1

        if step % 100 == 0:
            print(f"Step {step}/{max_steps}, Episode {episode_num}")

    # Save video
    if frames:
        print(f"Saving {len(frames)} frames...")
        imageio.mimsave(output_path, frames, fps=50)
        print(f"Video saved to {output_path}")
    else:
        print("No frames captured")


def main():
    """Run the demo."""
    parser = argparse.ArgumentParser(description="Render humanoid hug demo")
    parser.add_argument(
        "--mode",
        choices=["human", "video"],
        default="human",
        help="Rendering mode",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="humanoid_hug_demo.mp4",
        help="Output video path (for video mode)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Maximum steps to run",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=0,
        help="Curriculum stage (0-3)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Humanoid Hug Environment - Render Demo")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Max steps: {args.steps}")
    print(f"Stage: {args.stage}")
    print("=" * 60)

    if args.mode == "human":
        env = HumanoidHugEnv(
            render_mode="human",
            horizon=args.steps,
            stage=args.stage,
        )
        try:
            run_demo_human(env, args.steps)
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
        finally:
            env.close()
    else:
        env = HumanoidHugEnv(
            render_mode="rgb_array",
            horizon=args.steps,
            stage=args.stage,
        )
        try:
            run_demo_video(env, args.output, args.steps)
        finally:
            env.close()


if __name__ == "__main__":
    main()
