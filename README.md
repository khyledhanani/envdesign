# Humanoid Hug Environment

A two-agent humanoid hugging environment using MuJoCo and PettingZoo for multi-agent reinforcement learning research.

## Overview

This environment features two humanoid agents that must learn to:
1. **Approach** each other from a distance
2. **Align** face-to-face with proper orientation
3. **Embrace** by wrapping arms around each other's torso
4. **Hold** a stable hug for a sustained duration

The environment uses the PettingZoo Parallel API, making it suitable for training IPPO/MAPPO-style decentralized policies.

## Installation

```bash
# Clone and install in editable mode
cd humanoid_hug
pip install -e .

# For development (includes pytest)
pip install -e ".[dev]"

# For training with RLlib
pip install -e ".[train]"
```

### Dependencies

- `mujoco>=3.0.0` - Physics simulation
- `numpy>=1.21.0` - Numerical computation
- `pettingzoo>=1.24.0` - Multi-agent environment API
- `gymnasium>=0.29.0` - RL environment spaces

## Quick Start

### Run Random Rollout

```bash
python scripts/rollout_random.py --episodes 3 --horizon 500 --stage 0
```

### Render Demo

```bash
# Human viewer (requires display)
python scripts/render_demo.py --mode human --steps 500

# Save to video file
python scripts/render_demo.py --mode video --output demo.mp4 --steps 500
```

### Basic Usage in Code

```python
from humanoid_hug import HumanoidHugEnv

# Create environment
env = HumanoidHugEnv(
    render_mode=None,  # or "human" / "rgb_array"
    horizon=1000,
    stage=0,  # Curriculum stage (0-3)
)

# Reset
obs, infos = env.reset(seed=42)

# Run episode
while env.agents:
    # Get actions for each agent (from your policy)
    actions = {
        "h0": env.action_space("h0").sample(),
        "h1": env.action_space("h1").sample(),
    }

    obs, rewards, terminations, truncations, infos = env.step(actions)

env.close()
```

## Environment Details

### Agents

- **h0**: First humanoid (blue)
- **h1**: Second humanoid (orange)

Both agents have identical observation and action spaces.

### Observation Space

Each agent receives a flat float32 vector containing:

| Component | Description | Dimension |
|-----------|-------------|-----------|
| Joint positions | All joints except root position | ~17 |
| Joint velocities | All joints except root velocity | ~17 |
| Root quaternion | Torso orientation | 4 |
| Root angular velocity | Torso rotation rate | 3 |
| Partner chest position | Relative, in local frame | 3 |
| Partner pelvis position | Relative, in local frame | 3 |
| Partner velocity | Relative, in local frame | 3 |
| Facing alignment | Dot product of forward vectors | 1 |
| Up alignment | Dot product of up vectors | 1 |
| Contact bits | L-arm, R-arm, partner arm contacts | 3 |

### Action Space

Each agent controls 18 actuators:
- 2 abdomen joints (z, y)
- 7 left leg joints (hip x/y/z, knee, ankle)
- 7 right leg joints
- 3 left arm joints (shoulder1/2, elbow)
- 3 right arm joints

Actions are continuous in the range `[-1, 1]`.

### Hug Success Condition

A successful hug requires ALL of the following for `K` consecutive steps (default K=30):

1. **Distance**: Chest-to-chest distance between 0.25m and 0.60m
2. **Facing**: Forward vectors aligned for face-to-face (dot product > 0.6)
3. **Contact**: Both agents have arm-to-torso contact with the other
4. **Stability**: Relative velocity < 1.0 m/s
5. **Upright**: Both agents tilted less than ~30 degrees from vertical

### Reward Structure

The reward combines several components (weights vary by curriculum stage):

#### Shaping Rewards (encourage approach)
- **Distance**: `exp(-3.0 * chest_distance)` - Higher when closer
- **Facing**: `max(0, facing_alignment)` - Higher when facing each other
- **Stability**: `exp(-2.0 * relative_speed)` - Higher when both moving slowly

#### Embrace Rewards (encourage contact)
- **Contact**: +1 for each agent with arm-torso contact (max +2)
- **Hand-to-back**: Distance-based shaping for hands reaching partner's back

#### Penalties (prevent bad behavior)
- **Fall**: Large negative penalty if either humanoid falls
- **Energy**: Small penalty for control effort (`-0.001 * sum(ctrl^2)`)
- **Impact**: Small penalty for violent collisions

#### Success Bonus
- +100 to +200 when hug hold target is reached

### Termination

Episodes end when:
- **Success**: Hug held for K steps (positive termination)
- **Fall**: Either humanoid falls (negative termination)
- **NaN**: Numerical instability detected (negative termination)
- **Horizon**: Maximum steps reached (truncation)

## Curriculum Stages

Use `options={"stage": N}` in `reset()` to set curriculum stage:

| Stage | Focus | Description |
|-------|-------|-------------|
| 0 | Approach | Distance + facing rewards only |
| 1 | Reach | Add hand-to-back shaping |
| 2 | Contact | Add arm-torso contact reward |
| 3 | Full Hug | Full rewards, higher success bonus |

```python
# Start training at stage 0
obs, infos = env.reset(seed=42, options={"stage": 0})

# Progress to stage 2 when policy improves
obs, infos = env.reset(seed=43, options={"stage": 2})
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_env_api.py -v

# Run with coverage
pytest tests/ --cov=humanoid_hug --cov-report=term-missing
```

## Project Structure

```
humanoid_hug/
├── pyproject.toml          # Package configuration
├── README.md               # This file
├── humanoid_hug/
│   ├── __init__.py
│   ├── env.py              # Main environment class
│   ├── mjcf/
│   │   └── humanoid_hug.xml  # MuJoCo model
│   └── utils/
│       ├── ids.py          # ID caching for efficient lookups
│       ├── kinematics.py   # Coordinate transforms
│       ├── contacts.py     # Contact detection
│       ├── obs.py          # Observation builder
│       └── rewards.py      # Reward computation
├── scripts/
│   ├── rollout_random.py   # Random rollout demo
│   └── render_demo.py      # Rendering demo
└── tests/
    ├── test_env_api.py     # API conformance tests
    ├── test_obs_shapes.py  # Shape/dtype tests
    └── test_contact_logic.py  # Contact detection tests
```

## Training Tips

1. **Start with Stage 0**: Let agents learn to approach before requiring contact
2. **Progress gradually**: Move to higher stages once approach is stable
3. **Use frame skip**: Default is 5 physics steps per action
4. **Monitor hug_hold_steps**: This metric shows how close agents are to success
5. **Watch for falls**: High fall rate may indicate too aggressive actions

## Known Limitations

- Contact detection is binary (no force magnitude)
- Humanoid model is simplified (no fingers/detailed hands)
- Success requires perfect coordination between agents

## License

MIT License
