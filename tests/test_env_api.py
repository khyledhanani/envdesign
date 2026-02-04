"""Tests for the environment API conformance."""

import pytest
import numpy as np

from humanoid_hug import HumanoidHugEnv


class TestEnvAPI:
    """Test suite for environment API."""

    @pytest.fixture
    def env(self):
        """Create environment fixture."""
        env = HumanoidHugEnv(render_mode=None, horizon=100)
        yield env
        env.close()

    def test_reset_returns_dict_obs(self, env):
        """Test that reset returns dict observations for both agents."""
        obs, infos = env.reset(seed=42)

        assert isinstance(obs, dict)
        assert "h0" in obs
        assert "h1" in obs
        assert isinstance(obs["h0"], np.ndarray)
        assert isinstance(obs["h1"], np.ndarray)

    def test_reset_returns_dict_infos(self, env):
        """Test that reset returns dict infos for both agents."""
        obs, infos = env.reset(seed=42)

        assert isinstance(infos, dict)
        assert "h0" in infos
        assert "h1" in infos
        assert isinstance(infos["h0"], dict)
        assert isinstance(infos["h1"], dict)

    def test_step_returns_correct_dicts(self, env):
        """Test that step returns dicts with matching keys."""
        obs, infos = env.reset(seed=42)

        actions = {
            "h0": env.action_space("h0").sample(),
            "h1": env.action_space("h1").sample(),
        }

        obs, rewards, terminations, truncations, infos = env.step(actions)

        # Check observations
        assert isinstance(obs, dict)
        assert set(obs.keys()) == {"h0", "h1"} or set(obs.keys()) == set()

        # Check rewards
        assert isinstance(rewards, dict)
        for agent in rewards:
            assert isinstance(rewards[agent], (int, float))

        # Check terminations
        assert isinstance(terminations, dict)
        for agent in terminations:
            assert isinstance(terminations[agent], bool)

        # Check truncations
        assert isinstance(truncations, dict)
        for agent in truncations:
            assert isinstance(truncations[agent], bool)

        # Check infos
        assert isinstance(infos, dict)

    def test_100_random_steps_no_crash(self, env):
        """Test that 100 random steps don't crash."""
        obs, infos = env.reset(seed=42)

        for step in range(100):
            if not env.agents:
                break

            actions = {
                agent: env.action_space(agent).sample()
                for agent in env.agents
            }

            obs, rewards, terminations, truncations, infos = env.step(actions)

            # Basic sanity checks
            assert not np.any(np.isnan(list(rewards.values())))

    def test_agents_list_correct(self, env):
        """Test that agents list is correctly maintained."""
        obs, infos = env.reset(seed=42)

        assert env.agents == ["h0", "h1"]
        assert env.possible_agents == ["h0", "h1"]

    def test_action_space_valid(self, env):
        """Test that action spaces are valid."""
        for agent in env.possible_agents:
            space = env.action_space(agent)
            assert hasattr(space, "sample")
            assert hasattr(space, "shape")
            assert space.dtype == np.float32

    def test_observation_space_valid(self, env):
        """Test that observation spaces are valid."""
        for agent in env.possible_agents:
            space = env.observation_space(agent)
            assert hasattr(space, "sample")
            assert hasattr(space, "shape")
            assert space.dtype == np.float32

    def test_reset_with_seed_deterministic(self, env):
        """Test that reset with same seed gives same initial state."""
        obs1, _ = env.reset(seed=123)
        obs2, _ = env.reset(seed=123)

        np.testing.assert_array_equal(obs1["h0"], obs2["h0"])
        np.testing.assert_array_equal(obs1["h1"], obs2["h1"])

    def test_curriculum_stage_option(self, env):
        """Test that curriculum stage can be set via options."""
        obs, infos = env.reset(seed=42, options={"stage": 2})

        assert env.stage == 2
        assert infos["h0"]["stage"] == 2

    def test_horizon_truncation(self):
        """Test that episode truncates at horizon."""
        env = HumanoidHugEnv(render_mode=None, horizon=10)
        obs, infos = env.reset(seed=42)

        for _ in range(20):
            if not env.agents:
                break
            actions = {agent: np.zeros(env.action_space(agent).shape) for agent in env.agents}
            obs, rewards, terminations, truncations, infos = env.step(actions)

        # Should have ended (either truncated or terminated)
        assert len(env.agents) == 0
        env.close()
