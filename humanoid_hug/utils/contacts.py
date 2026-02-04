"""Contact detection utilities for arm-torso hug detection."""

from typing import Dict, Set
import numpy as np
import mujoco

from humanoid_hug.utils.ids import IDCache


class ContactDetector:
    """Detects contacts between humanoid arms and torsos for hug detection."""

    def __init__(self, id_cache: IDCache):
        """Initialize the contact detector.

        Args:
            id_cache: Cached IDs from the MuJoCo model
        """
        self.id_cache = id_cache

    def detect_contacts(self, data: mujoco.MjData) -> Dict[str, bool]:
        """Detect arm-to-torso contacts for hug evaluation.

        Iterates through all contacts and checks if one humanoid's arm
        geoms are in contact with the other humanoid's torso geoms.

        Args:
            data: MuJoCo data instance

        Returns:
            Dictionary with contact detection results:
                - h0_arm_torso: True if any h0 arm touches h1 torso
                - h1_arm_torso: True if any h1 arm touches h0 torso
                - h0_l_arm_torso: True if h0 left arm touches h1 torso
                - h0_r_arm_torso: True if h0 right arm touches h1 torso
                - h1_l_arm_torso: True if h1 left arm touches h0 torso
                - h1_r_arm_torso: True if h1 right arm touches h0 torso
                - any_arm_torso: True if any arm-torso contact exists
        """
        result = {
            "h0_arm_torso": False,
            "h1_arm_torso": False,
            "h0_l_arm_torso": False,
            "h0_r_arm_torso": False,
            "h1_l_arm_torso": False,
            "h1_r_arm_torso": False,
            "any_arm_torso": False,
        }

        h0_arms = self.id_cache.arm_geoms["h0"]
        h1_arms = self.id_cache.arm_geoms["h1"]
        h0_torso = self.id_cache.torso_geoms["h0"]
        h1_torso = self.id_cache.torso_geoms["h1"]

        h0_l_arm = self.id_cache.left_arm_geoms["h0"]
        h0_r_arm = self.id_cache.right_arm_geoms["h0"]
        h1_l_arm = self.id_cache.left_arm_geoms["h1"]
        h1_r_arm = self.id_cache.right_arm_geoms["h1"]

        # Iterate through active contacts
        for i in range(data.ncon):
            contact = data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2

            # Check h0 arm touching h1 torso
            if (geom1 in h0_arms and geom2 in h1_torso) or \
               (geom2 in h0_arms and geom1 in h1_torso):
                result["h0_arm_torso"] = True

                # Check specific arm
                if (geom1 in h0_l_arm and geom2 in h1_torso) or \
                   (geom2 in h0_l_arm and geom1 in h1_torso):
                    result["h0_l_arm_torso"] = True
                if (geom1 in h0_r_arm and geom2 in h1_torso) or \
                   (geom2 in h0_r_arm and geom1 in h1_torso):
                    result["h0_r_arm_torso"] = True

            # Check h1 arm touching h0 torso
            if (geom1 in h1_arms and geom2 in h0_torso) or \
               (geom2 in h1_arms and geom1 in h0_torso):
                result["h1_arm_torso"] = True

                # Check specific arm
                if (geom1 in h1_l_arm and geom2 in h0_torso) or \
                   (geom2 in h1_l_arm and geom1 in h0_torso):
                    result["h1_l_arm_torso"] = True
                if (geom1 in h1_r_arm and geom2 in h0_torso) or \
                   (geom2 in h1_r_arm and geom1 in h0_torso):
                    result["h1_r_arm_torso"] = True

        result["any_arm_torso"] = result["h0_arm_torso"] or result["h1_arm_torso"]

        return result

    def get_contact_force_proxy(self, data: mujoco.MjData) -> float:
        """Get a simple proxy for total contact force magnitude.

        This is a rough estimate using the number of contacts and
        relative velocities at contact points.

        Args:
            data: MuJoCo data instance

        Returns:
            Proxy for contact force (higher = more forceful contacts)
        """
        if data.ncon == 0:
            return 0.0

        total_force_proxy = 0.0
        for i in range(data.ncon):
            contact = data.contact[i]
            # Use contact depth as a force proxy
            if contact.dist < 0:  # Penetration depth
                total_force_proxy += abs(contact.dist) * 1000  # Scale up

        return total_force_proxy

    def check_floor_contact(self, data: mujoco.MjData, model: mujoco.MjModel) -> Dict[str, bool]:
        """Check if humanoid bodies (other than feet) are touching the floor.

        Args:
            data: MuJoCo data instance
            model: MuJoCo model instance

        Returns:
            Dictionary with floor contact status for each agent
        """
        result = {"h0_floor": False, "h1_floor": False}

        # Get floor geom ID
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor_id < 0:
            return result

        # Foot geoms are allowed to touch floor
        h0_feet = set()
        h1_feet = set()
        for i in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name is None:
                continue
            if "foot" in name.lower():
                if name.startswith("h0_"):
                    h0_feet.add(i)
                elif name.startswith("h1_"):
                    h1_feet.add(i)

        # Get all geoms for each humanoid
        h0_geoms = set()
        h1_geoms = set()
        for i in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name is None:
                continue
            if name.startswith("h0_"):
                h0_geoms.add(i)
            elif name.startswith("h1_"):
                h1_geoms.add(i)

        # Check contacts
        for i in range(data.ncon):
            contact = data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2

            floor_in_contact = (geom1 == floor_id or geom2 == floor_id)
            if not floor_in_contact:
                continue

            other_geom = geom2 if geom1 == floor_id else geom1

            # Check if non-foot h0 geom touches floor
            if other_geom in h0_geoms and other_geom not in h0_feet:
                result["h0_floor"] = True

            # Check if non-foot h1 geom touches floor
            if other_geom in h1_geoms and other_geom not in h1_feet:
                result["h1_floor"] = True

        return result
