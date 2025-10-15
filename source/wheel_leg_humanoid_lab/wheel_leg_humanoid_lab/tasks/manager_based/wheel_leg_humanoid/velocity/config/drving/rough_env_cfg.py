# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import isaaclab.terrains as terrain_gen
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
# from wheel_leg_humanoid_lab.assets import WHEEL_LEG_HUMANOID_DRIVING_MODE_CFG  # isort: skip
from wheel_leg_humanoid_lab.assets import WHEEL_LEG_HUMANOID_CFG
import numpy as np

# use other terrain
ROUGH_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.4,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.5, noise_range=(0.01, 0.05), noise_step=0.02, border_width=0.25
        ),
    },
)

@configclass
class WheelLegHumanoidDrivingActionsCfg(ActionsCfg):
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[""], scale=0.25, use_default_offset=True, clip=None, preserve_order=True
    )

    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=[""], scale=5.0, use_default_offset=True, clip=None, preserve_order=True
    )


@configclass
class WheelLegHumanoidDrivingRewardsCfg(RewardsCfg):
    """Reward terms for the MDP."""

    joint_vel_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    joint_acc_wheel_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    joint_torques_wheel_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )


@configclass
class WheelLegHumanoidDrivingRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    actions: WheelLegHumanoidDrivingActionsCfg = WheelLegHumanoidDrivingActionsCfg()
    rewards: WheelLegHumanoidDrivingRewardsCfg = WheelLegHumanoidDrivingRewardsCfg()

    base_link_name = "base_link"
    foot_link_name = ".*_foot_link" # foot_link is contained in knee_link
    wheel_link_name = [
        "left_wheel", "right_wheel",
        "left_foot_wheel_L",  "left_foot_wheel_R", "right_foot_wheel_L", "right_foot_wheel_R"
    ]

    # fmt: off
    body_joint_names = [
        "torso_joint"
    ]
    leg_joint_names = [
        # left
        "left_hip_roll_joint", "left_hip_yaw_joint", "left_hip_pitch_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint", "left_ankle_roll_joint",

        # right
        "right_hip_roll_joint", "right_hip_yaw_joint", "right_hip_pitch_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint", "right_ankle_roll_joint",
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
        self.scene.robot = WHEEL_LEG_HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.53)
        self.scene.robot.init_state.joint_pos.update({
            "right_pelvis_1": np.deg2rad(-20.0),
            "left_pelvis_1": np.deg2rad(-20.0),
            "right_calf": np.deg2rad(-120.0),
            "left_calf": np.deg2rad(-120.0),
        })
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = ROUGH_ROAD_CFG


        # ------------------------------Observations------------------------------
        self.observations.policy.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.policy.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        self.observations.critic.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.critic.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = {
            ".*_hip_roll_joint": 0.125, "^(?!.*_hip_roll_joint).*": 0.25
        }
        self.actions.joint_vel.scale = 5.0
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_vel.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.leg_joint_names
        self.actions.joint_vel.joint_names = self.wheel_joint_names

        # ------------------------------Events------------------------------
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.1),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        # ------------------------------Rewards------------------------------
        # General
        self.rewards.is_terminated.weight = -200

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -0.1
        self.rewards.base_height_l2.weight = -2.0
        self.rewards.base_height_l2.params["target_height"] = 0.45
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_torques_wheel_l2.weight = 0
        self.rewards.joint_torques_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_wheel_l2.weight = 0
        self.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_acc_wheel_l2.weight = -2.5e-9
        self.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        # self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -0.2, [".*_hip_joint"])
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_limits.weight = 0.0
        self.rewards.joint_vel_limits.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_power.weight = -2e-5
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.stand_still.weight = -2.0
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.leg_joint_names + self.body_joint_names
        self.rewards.joint_pos_penalty.weight = -1.0
        self.rewards.joint_pos_penalty.params["asset_cfg"].joint_names = self.leg_joint_names
        # self.rewards.joint_pos_penalty.params["velocity_threshold"] = 100
        self.rewards.wheel_vel_penalty.weight = -0.01
        self.rewards.wheel_vel_penalty.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.wheel_vel_penalty.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_mirror.weight = -0.05
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["left_hip_roll_joint", "right_hip_roll_joint"],
            ["left_hip_yaw_joint", "right_hip_yaw_joint"],
            ["left_hip_pitch_joint", "right_hip_pitch_joint"],
            ["left_knee_joint", "right_knee_joint"],
            ["left_ankle_pitch_joint", "right_ankle_pitch_joint"],
            ["left_ankle_roll_joint", "right_ankle_roll_joint"],
        ]

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.01

        # Contact sensor
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!({'|'.join(self.wheel_link_name)})).*"]
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = self.wheel_link_name

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5

        # Others
        self.rewards.feet_air_time.weight = 0
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = self.wheel_link_name
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = self.wheel_link_name
        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = self.wheel_link_name
        self.rewards.feet_stumble.weight = -0.1
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = self.wheel_link_name
        self.rewards.feet_slide.weight = -0.1
        self.rewards.feet_slide.params["sensor_cfg"].body_names = self.wheel_link_name
        self.rewards.feet_slide.params["asset_cfg"].body_names = self.wheel_link_name
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height.params["target_height"] = 0.1
        self.rewards.feet_height.params["asset_cfg"].body_names = self.wheel_link_name
        self.rewards.feet_height_body.weight = 0
        self.rewards.feet_height_body.params["target_height"] = -0.2
        self.rewards.feet_height_body.params["asset_cfg"].body_names = self.wheel_link_name
        self.rewards.feet_gait.weight = 0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("left_wheel", "right_wheel"), ("left_foot_wheel_L", "right_foot_wheel_R"), ("left_foot_wheel_R", "right_foot_wheel_L"))
        self.rewards.upward.weight = 1.0
        self.rewards.feet_distance_y_exp.weight = -1.0
        self.rewards.feet_distance_y_exp.params["stance_width"] = 0.6
        self.rewards.feet_distance_y_exp.params["asset_cfg"].body_names = self.wheel_link_name

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "WheelLegHumanoidDrivingRoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = "torso_link"

        # ------------------------------Curriculums------------------------------
        # self.curriculum.command_levels.params["range_multiplier"] = (0.2, 1.0)
        self.curriculum.command_levels = None

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
