"""Tests for contact detection logic."""

import pytest
import numpy as np

from humanoid_hug import HumanoidHugEnv
from humanoid_hug.utils.contacts import ContactDetector
from humanoid_hug.utils.ids import IDCache


class TestContactLogic:
    """Test suite for contact detection."""

    @pytest.fixture
    def env(self):
        """Create environment fixture."""
        env = HumanoidHugEnv(render_mode=None, horizon=100)
        yield env
        env.close()

    def test_contact_detector_init(self, env):
        """Test that contact detector initializes correctly."""
        detector = ContactDetector(env.id_cache)
        assert detector is not None
        assert detector.id_cache is not None

    def test_detect_contacts_returns_dict(self, env):
        """Test that detect_contacts returns proper dict."""
        env.reset(seed=42)
        detector = ContactDetector(env.id_cache)

        contact_info = detector.detect_contacts(env.data)

        assert isinstance(contact_info, dict)
        assert "h0_arm_torso" in contact_info
        assert "h1_arm_torso" in contact_info
        assert "h0_l_arm_torso" in contact_info
        assert "h0_r_arm_torso" in contact_info
        assert "h1_l_arm_torso" in contact_info
        assert "h1_r_arm_torso" in contact_info
        assert "any_arm_torso" in contact_info

    def test_contact_values_are_bool(self, env):
        """Test that contact values are booleans."""
        env.reset(seed=42)
        detector = ContactDetector(env.id_cache)

        contact_info = detector.detect_contacts(env.data)

        for key, value in contact_info.items():
            assert isinstance(value, bool), f"{key} is not bool: {type(value)}"

    def test_contact_logic_runs_multiple_steps(self, env):
        """Test that contact logic runs without error for multiple steps."""
        env.reset(seed=42)
        detector = ContactDetector(env.id_cache)

        for step in range(50):
            if not env.agents:
                break

            contact_info = detector.detect_contacts(env.data)

            # Verify all expected keys present
            expected_keys = [
                "h0_arm_torso", "h1_arm_torso",
                "h0_l_arm_torso", "h0_r_arm_torso",
                "h1_l_arm_torso", "h1_r_arm_torso",
                "any_arm_torso",
            ]
            for key in expected_keys:
                assert key in contact_info

            # Step the environment
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            env.step(actions)

    def test_contact_consistency(self, env):
        """Test that any_arm_torso is consistent with individual contacts."""
        env.reset(seed=42)
        detector = ContactDetector(env.id_cache)

        for step in range(50):
            if not env.agents:
                break

            contact_info = detector.detect_contacts(env.data)

            # any_arm_torso should be True if either h0 or h1 arm touches torso
            expected_any = contact_info["h0_arm_torso"] or contact_info["h1_arm_torso"]
            assert contact_info["any_arm_torso"] == expected_any, \
                f"Step {step}: any_arm_torso inconsistent"

            # If h0_arm_torso is True, at least one of l/r should be True
            if contact_info["h0_arm_torso"]:
                assert contact_info["h0_l_arm_torso"] or contact_info["h0_r_arm_torso"], \
                    f"Step {step}: h0_arm_torso True but no l/r"

            if contact_info["h1_arm_torso"]:
                assert contact_info["h1_l_arm_torso"] or contact_info["h1_r_arm_torso"], \
                    f"Step {step}: h1_arm_torso True but no l/r"

            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            env.step(actions)

    def test_id_cache_geom_sets(self, env):
        """Test that ID cache has proper geom sets."""
        id_cache = env.id_cache

        # Both agents should have arm and torso geom sets
        for agent in ["h0", "h1"]:
            assert agent in id_cache.arm_geoms
            assert agent in id_cache.torso_geoms
            assert agent in id_cache.left_arm_geoms
            assert agent in id_cache.right_arm_geoms

            # Should have some geoms in each set
            assert len(id_cache.arm_geoms[agent]) > 0, f"{agent} has no arm geoms"
            assert len(id_cache.torso_geoms[agent]) > 0, f"{agent} has no torso geoms"

    def test_contact_force_proxy(self, env):
        """Test that contact force proxy returns valid value."""
        env.reset(seed=42)
        detector = ContactDetector(env.id_cache)

        for step in range(20):
            if not env.agents:
                break

            force_proxy = detector.get_contact_force_proxy(env.data)

            assert isinstance(force_proxy, float)
            assert not np.isnan(force_proxy)
            assert force_proxy >= 0

            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            env.step(actions)

    def test_floor_contact_detection(self, env):
        """Test floor contact detection."""
        env.reset(seed=42)
        detector = ContactDetector(env.id_cache)

        floor_contact = detector.check_floor_contact(env.data, env.model)

        assert isinstance(floor_contact, dict)
        assert "h0_floor" in floor_contact
        assert "h1_floor" in floor_contact
        assert isinstance(floor_contact["h0_floor"], bool)
        assert isinstance(floor_contact["h1_floor"], bool)
