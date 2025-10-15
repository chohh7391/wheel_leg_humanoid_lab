# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment (similar to OpenAI Gym Ant-v2).
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

'''Driving Rough'''
gym.register(
    id="Wheel-Leg-Humanoid-Velocity-Driving-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:WheelLegHumanoidDrivingRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegHumanoidDrivingRoughPPORunnerCfg",
    },
)

gym.register(
    id="Wheel-Leg-Humanoid-Velocity-Driving-Rough-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_fixed_hip_cfg:WheelLegHumanoidDrivingFixedHipRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegHumanoidDrivingFixedHipRoughPPORunnerCfg",
    },
)

gym.register(
    id="Wheel-Leg-Humanoid-Velocity-Driving-Rough-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_fixed_hip_cfg:WheelLegHumanoidDrivingFixedHipRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegHumanoidDrivingFixedHipRoughPPORunnerCfg",
    },
)

gym.register(
    id="Wheel-Leg-Humanoid-Velocity-Driving-Rough-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_fixed_ankle_cfg:WheelLegHumanoidDrivingFixedAnkleRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegHumanoidDrivingFixedAnkleRoughPPORunnerCfg",
    },
)

gym.register(
    id="Wheel-Leg-Humanoid-Velocity-Driving-Rolling-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_rolling_cfg:WheelLegHumanoidDrivingRollingRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegHumanoidDrivingRollingRoughPPORunnerCfg",
    },
)




'''Driving Flat'''
gym.register(
    id="Wheel-Leg-Humanoid-Velocity-Driving-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:WheelLegHumanoidDrivingFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegHumanoidDrivingFlatPPORunnerCfg",
    },
)

gym.register(
    id="Wheel-Leg-Humanoid-Velocity-Driving-Flat-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:WheelLegHumanoidDrivingFixedHipFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegHumanoidDrivingFixedHipFlatPPORunnerCfg",
    },
)

gym.register(
    id="Wheel-Leg-Humanoid-Velocity-Driving-Flat-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:WheelLegHumanoidDrivingFixedAnkleFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegHumanoidDrivingFixedAnkleFlatPPORunnerCfg",
    },
)

gym.register(
    id="Wheel-Leg-Humanoid-Velocity-Driving-Flat-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:WheelLegHumanoidDrivingFixedHipAnkleFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegHumanoidDrivingFixedHipAnkleFlatPPORunnerCfg",
    },
)

gym.register(
    id="Wheel-Leg-Humanoid-Velocity-Driving-Rolling-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:WheelLegHumanoidDrivingRollingFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegHumanoidDrivingRollingFlatPPORunnerCfg",
    },
)