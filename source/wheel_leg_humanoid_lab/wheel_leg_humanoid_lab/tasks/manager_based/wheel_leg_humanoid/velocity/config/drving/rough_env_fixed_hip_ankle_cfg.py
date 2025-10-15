# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import wheel_leg_humanoid_lab.tasks.manager_based.wheel_leg_humanoid.velocity.mdp as mdp
from wheel_leg_humanoid_lab.tasks.manager_based.wheel_leg_humanoid.velocity.velocity_env_cfg import (
    ActionsCfg,
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

##
# Pre-defined configs
##
from wheel_leg_humanoid_lab.assets import WHEEL_LEG_HUMANOID_DRIVING_MODE_FIXED_HIP_ANKLE_CFG  # isort: skip
from .rough_env_cfg import WheelLegHumanoidDrivingRoughEnvCfg

@configclass
class WheelLegHumanoidDrivingFixedHipAnkleRoughEnvCfg(WheelLegHumanoidDrivingRoughEnvCfg):

    # link names
    foot_link_name = ".*_knee_link" # foot link is contained in knee link, because ankle joint is fixed

    # fmt: off
    body_joint_names = [
        "torso_joint"
    ]
    leg_joint_names = [
        # left
        # "left_hip_roll_joint", "left_hip_yaw_joint", "left_hip_pitch_joint",
        "left_knee_joint",
        # "left_ankle_pitch_joint", "left_ankle_roll_joint",

        # right
        # "right_hip_roll_joint", "right_hip_yaw_joint", "right_hip_pitch_joint",
        "right_knee_joint",
        # "right_ankle_pitch_joint", "right_ankle_roll_joint",
    ]
    wheel_joint_names = [
        "left_wheel_joint", "right_wheel_joint",
    ]
    joint_names = body_joint_names + leg_joint_names + wheel_joint_names
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        self.scene.robot = WHEEL_LEG_HUMANOID_DRIVING_MODE_FIXED_HIP_ANKLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # ------------------------------Observations------------------------------
        # equal to parent

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = {
            ".*": 0.25 # hip roll joint is fixed -> no need to control
        }
        self.actions.joint_vel.scale = 5.0
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_vel.clip = {".*": (-100.0, 100.0)}

        # ------------------------------Events------------------------------
        # equal to parent

        # ------------------------------Rewards------------------------------
        # General
        self.rewards.is_terminated.weight = -200

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -0.1
        self.rewards.base_height_l2.weight = -0.5
        self.rewards.base_height_l2.params["target_height"] = 0.45
        self.rewards.body_lin_acc_l2.weight = 0


        # Joint penalties
        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.joint_torques_wheel_l2.weight = 0
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_vel_wheel_l2.weight = 0
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_acc_wheel_l2.weight = -2.5e-9
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_vel_limits.weight = 0.0
        self.rewards.joint_power.weight = -2e-5
        self.rewards.stand_still.weight = -2.0
        self.rewards.joint_pos_penalty.weight = -1.0
        # self.rewards.joint_pos_penalty.params["velocity_threshold"] = 100
        self.rewards.wheel_vel_penalty.weight = -0.01
        self.rewards.joint_mirror.weight = -0.05
        self.rewards.joint_mirror.params["mirror_joints"] = [
            # ["left_hip_roll_joint", "right_hip_roll_joint"],
            # ["left_hip_yaw_joint", "right_hip_yaw_joint"],
            # ["left_hip_pitch_joint", "right_hip_pitch_joint"],
            ["left_knee_joint", "right_knee_joint"],
            # ["left_ankle_pitch_joint", "right_ankle_pitch_joint"],
            # ["left_ankle_roll_joint", "right_ankle_roll_joint"],
        ]

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.01

        # Contact sensor
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.contact_forces.weight = -1.5e-4

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5

        # Others
        self.rewards.feet_air_time.weight = 0
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_stumble.weight = -0.1
        self.rewards.feet_slide.weight = -0.1
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height.params["target_height"] = 0.1
        self.rewards.feet_height_body.weight = 0
        self.rewards.feet_height_body.params["target_height"] = -0.2
        self.rewards.feet_gait.weight = 0
        self.rewards.upward.weight = 1.0
        self.rewards.feet_distance_y_exp.weight = -1.0
        self.rewards.feet_distance_y_exp.params["stance_width"] = 0.6

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "WheelLegHumanoidDrivingFixedHipAnkleRoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        # equal to parent

        # ------------------------------Curriculums------------------------------
        # equal to parent

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
