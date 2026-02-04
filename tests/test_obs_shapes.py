"""Tests for observation and action shapes."""

import pytest
import numpy as np

from humanoid_hug import HumanoidHugEnv


class TestObsShapes:
    """Test suite for observation and action shapes."""

    @pytest.fixture
    def env(self):
        """Create environment fixture."""
        env = HumanoidHugEnv(render_mode=None, horizon=100)
        yield env
        env.close()

    def test_obs_shape_matches_space(self, env):
        """Test that observation shape matches observation space."""
        obs, _ = env.reset(seed=42)

        for agent in env.agents:
            expected_shape = env.observation_space(agent).shape
            actual_shape = obs[agent].shape
            assert actual_shape == expected_shape, \
                f"Agent {agent}: expected {expected_shape}, got {actual_shape}"

    def test_obs_dtype_correct(self, env):
        """Test that observation dtype is float32."""
        obs, _ = env.reset(seed=42)

        for agent in env.agents:
            assert obs[agent].dtype == np.float32, \
                f"Agent {agent}: expected float32, got {obs[agent].dtype}"

    def test_obs_shape_consistent_across_steps(self, env):
        """Test that observation shape is consistent across steps."""
        obs, _ = env.reset(seed=42)
        initial_shapes = {agent: obs[agent].shape for agent in env.agents}

        for step in range(50):
            if not env.agents:
                break

            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            obs, _, _, _, _ = env.step(actions)

            for agent in obs:
                assert obs[agent].shape == initial_shapes[agent], \
                    f"Step {step}, agent {agent}: shape changed"

    def test_action_space_matches_actuators(self, env):
        """Test that action space size matches number of actuators."""
        # Both agents should have the same number of actuators
        h0_action_dim = env.action_space("h0").shape[0]
        h1_action_dim = env.action_space("h1").shape[0]

        assert h0_action_dim == h1_action_dim, \
            f"Asymmetric action dims: h0={h0_action_dim}, h1={h1_action_dim}"

        # Should be 18 actuators per humanoid (based on our MJCF)
        assert h0_action_dim == 18, f"Expected 18 actuators, got {h0_action_dim}"

    def test_action_bounds_respected(self, env):
        """Test that action space bounds are [-1, 1]."""
        for agent in env.possible_agents:
            space = env.action_space(agent)
            np.testing.assert_array_equal(space.low, -np.ones(space.shape))
            np.testing.assert_array_equal(space.high, np.ones(space.shape))

    def test_obs_no_nan(self, env):
        """Test that observations don't contain NaN."""
        obs, _ = env.reset(seed=42)

        for agent in env.agents:
            assert not np.any(np.isnan(obs[agent])), \
                f"NaN in initial observation for {agent}"

        for step in range(50):
            if not env.agents:
                break

            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            obs, _, _, _, _ = env.step(actions)

            for agent in obs:
                assert not np.any(np.isnan(obs[agent])), \
                    f"NaN in observation at step {step} for {agent}"

    def test_obs_no_inf(self, env):
        """Test that observations don't contain infinity."""
        obs, _ = env.reset(seed=42)

        for agent in env.agents:
            assert not np.any(np.isinf(obs[agent])), \
                f"Inf in initial observation for {agent}"

        for step in range(50):
            if not env.agents:
                break

            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            obs, _, _, _, _ = env.step(actions)

            for agent in obs:
                assert not np.any(np.isinf(obs[agent])), \
                    f"Inf in observation at step {step} for {agent}"

    def test_obs_reasonable_values(self, env):
        """Test that observations have reasonable values (not too extreme)."""
        obs, _ = env.reset(seed=42)

        for agent in env.agents:
            # Most values should be within a reasonable range
            max_val = np.max(np.abs(obs[agent]))
            assert max_val < 1000, \
                f"Extreme observation value {max_val} for {agent}"
