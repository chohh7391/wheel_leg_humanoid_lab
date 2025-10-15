import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

import os

ROBOT_DESCRIPTION_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "kimm_wheel_legged_robot_description"
)

BODY_JOINTS = ["torso_joint"] # if base parts are added, this term should include them
HIP_JOINTS = [
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_hip_pitch_joint",
]
KNEE_JOINTS = [
    "left_knee_joint",
    "right_knee_joint",
]
ANKLE_JOINTS = [
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]
WHEEL_JOINTS = ["left_wheel_joint", "right_wheel_joint"]

WALKING_MODE_HEIGHT = 0.775

# Robot Configurations
WHEEL_LEG_HUMANOID_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ROBOT_DESCRIPTION_PATH}/usd/kimm_wheel_legged_robot/kimm_wheel_legged_robot.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, WALKING_MODE_HEIGHT),
        joint_pos={
            # Body
            "torso_joint": 0.0,

            # Left leg
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_hip_pitch_joint": 0.0,
            "left_knee_joint": 0.0,
            "left_ankle_pitch_joint": 0.0,
            "left_ankle_roll_joint": 0.0,
            "left_foot_wheel_L_joint": 0.0,
            "left_foot_wheel_R_joint": 0.0,
            "left_wheel_joint": 0.0,

            # Right leg
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_hip_pitch_joint": 0.0,
            "right_knee_joint": 0.0,
            "right_ankle_pitch_joint": 0.0,
            "right_ankle_roll_joint": 0.0,
            "right_foot_wheel_L_joint": 0.0,
            "right_foot_wheel_R_joint": 0.0,
            "right_wheel_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=BODY_JOINTS,
            effort_limit_sim=120.0,
            velocity_limit_sim=5.02,
            stiffness=200.0,
            damping=5.0,
            friction=0.0,
        ),
        "hips": ImplicitActuatorCfg(
            joint_names_expr=HIP_JOINTS,
            effort_limit_sim=120.0,
            velocity_limit_sim=5.02,
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
            },
            friction=0.0,
        ),
        "knees": ImplicitActuatorCfg(
            joint_names_expr=KNEE_JOINTS,
            effort_limit_sim=120.0,
            velocity_limit_sim=5.02,
            stiffness=200.0,
            damping=5.0,
            friction=0.0,
        ),
        "ankles": ImplicitActuatorCfg(
            joint_names_expr=ANKLE_JOINTS,
            effort_limit_sim=24.8,
            velocity_limit_sim=10.8,
            stiffness=100.0,
            damping=10.0,
            friction=0.0,
        ),
        # "wheels": ImplicitActuatorCfg(
        #     joint_names_expr=WHEEL_JOINTS,
        #     effort_limit_sim=120.0,
        #     velocity_limit_sim=5.02,
        #     stiffness=0.0,
        #     damping=0.5,
        #     friction=0.0,
        # ),
        # foot wheels are passive -> free joints
    }
)