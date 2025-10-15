# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .rough_env_cfg import WheelLegHumanoidDrivingRoughEnvCfg
from .rough_env_fixed_hip_cfg import WheelLegHumanoidDrivingFixedHipRoughEnvCfg
from .rough_env_fixed_ankle_cfg import WheelLegHumanoidDrivingFixedAnkleRoughEnvCfg
from .rough_env_fixed_hip_ankle_cfg import WheelLegHumanoidDrivingFixedHipAnkleRoughEnvCfg
from .rough_env_rolling_cfg import WheelLegHumanoidDrivingRollingRoughEnvCfg


@configclass
class WheelLegHumanoidDrivingFlatEnvCfg(WheelLegHumanoidDrivingRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "WheelLegHumanoidDrivingFlatEnvCfg":
            self.disable_zero_weight_rewards()

@configclass
class WheelLegHumanoidDrivingFixedHipFlatEnvCfg(WheelLegHumanoidDrivingFixedHipRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "WheelLegHumanoidDrivingFixedHipFlatEnvCfg":
            self.disable_zero_weight_rewards()

@configclass
class WheelLegHumanoidDrivingFixedAnkleFlatEnvCfg(WheelLegHumanoidDrivingFixedAnkleRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "WheelLegHumanoidDrivingFixedAnkleFlatEnvCfg":
            self.disable_zero_weight_rewards()

@configclass
class WheelLegHumanoidDrivingFixedHipAnkleFlatEnvCfg(WheelLegHumanoidDrivingFixedHipAnkleRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "WheelLegHumanoidDrivingFixedHipAnkleFlatEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class WheelLegHumanoidDrivingRollingFlatEnvCfg(WheelLegHumanoidDrivingRollingRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "WheelLegHumanoidDrivingRollingFlatEnvCfg":
            self.disable_zero_weight_rewards()