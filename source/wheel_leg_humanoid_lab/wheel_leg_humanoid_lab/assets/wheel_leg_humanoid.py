import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

import os

ROBOT_DESCRIPTION_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "wheel_leg_humanoid_description"
)

# Robot Configurations
WHEEL_LEG_HUMANOID_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ROBOT_DESCRIPTION_PATH}/urdf/wheel_leg_humanoid.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=3.0,
            max_angular_velocity=3.0,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.88),
        joint_pos={
            # waist
            "waist": 0.0,

            # pelvis
            "right_pelvis_1": 0.0,
            "right_pelvis_2": 0.0,
            "left_pelvis_1": 0.0,
            "left_pelvis_2": 0.0,

            # thigh
            "right_thigh": 0.0,
            "left_thigh": 0.0,

            # calf
            "right_calf": 0.0,
            "left_calf": 0.0,

            # ankle
            "right_ankle_1": 0.0,
            "right_ankle_2": 0.0,
            "left_ankle_1": 0.0,
            "left_ankle_2": 0.0,

            # wheel
            "right_wheel": 0.0,
            "left_wheel": 0.0,

            # foot wheel (passive)
            "right_foot_wheel_R": 0.0,
            "right_foot_wheel_L": 0.0,
            "left_foot_wheel_R": 0.0,
            "left_foot_wheel_L": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist"],
            effort_limit_sim=120.0,
            velocity_limit_sim=5.02,
            friction=0.0,
            stiffness={
                "waist": 200.0,
            },
            damping={
                "waist": 5.0,
            },
            armature={
                "waist": 0.01,
            },
        ),
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_pelvis_1", ".*_pelvis_2", ".*_thigh", ".*_calf"],
            effort_limit_sim=120.0,
            velocity_limit_sim=5.02,
            stiffness={
                ".*_pelvis_1": 200.0,
                ".*_pelvis_2": 150.0,
                ".*_thigh": 200.0,
                ".*_calf": 200.0,
            },
            damping={
                ".*_pelvis_1": 5.0,
                ".*_pelvis_2": 5.0,
                ".*_thigh": 5.0,
                ".*_calf": 5.0,
            },
            armature={
                ".*_pelvis_1": 0.01,
                ".*_pelvis_2": 0.01,
                ".*_thigh": 0.01,
                ".*_calf": 0.01,
            },
            friction=0.0,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_1", ".*_ankle_2"],
            effort_limit_sim=24.0,
            velocity_limit_sim=10.8,
            stiffness={
                ".*_ankle_1": 20.0,
                ".*_ankle_2": 20.0,
            },
            damping={
                ".*_ankle_1": 2.0,
                ".*_ankle_2": 2.0,
            },
            armature={
                ".*_ankle_1": 0.01,
                ".*_ankle_2": 0.01,
            },
            friction=0.0,
        ),
        "wheel": ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel"],
            effort_limit_sim=120.0,
            velocity_limit_sim=5.02,
            stiffness={
                ".*_wheel": 0.0,
            },
            damping={
                ".*_wheel": 0.5,
            },
            armature={
                ".*_wheel": 0.01,
            },
            friction=0.0,
        ),
        # foot wheel -> passive joint
    }
)