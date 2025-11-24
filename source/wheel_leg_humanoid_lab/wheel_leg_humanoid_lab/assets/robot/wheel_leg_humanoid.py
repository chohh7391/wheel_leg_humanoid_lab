import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from wheel_leg_humanoid_lab.assets import ISAACLAB_ASSETS_DATA_DIR
import numpy as np

"""
[CubeMars] AK80-64

Peak Torque (Nm): 120

Rated voltage (V): 24/48
Rated torque (Nm): 48
Rated speed (rpm): 23/48
"""

"""
[CubeMars] AK70-10

Peak Torque (Nm): 24.8

Rated voltage (V): 24/48
Rated torque (Nm): 8.3
Rated speed: 148/310 rpm
"""

def rpm2rad_per_s(rpm):

    rad_per_s = rpm * 2 * np.pi / 60
    
    return rad_per_s

# AK80-64
AK80_64 = {
    "peak_torque": 120,
    "rated_torque": 48,
    "rated_speed": rpm2rad_per_s(48)
}
AK70_10 = {
    "peak_torque": 24.8,
    "rated_torque": 8.3,
    "rated_speed": rpm2rad_per_s(310)
}

# WHEEL_RADIUS = 0.11
# L = 0.54

# MAX_LINVEL = WHEEL_RADIUS * AK80_64["rated_speed"]
# MAX_ANGVEL = MAX_LINVEL / L

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

# Robot Configurations
WHEEL_LEG_HUMANOID_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/kimm/wheel_leg_humanoid_description/urdf/wheel_leg_humanoid.urdf",
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
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
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
            effort_limit_sim=AK80_64["peak_torque"],
            velocity_limit_sim=AK80_64["rated_speed"],
            stiffness={
                "waist": STIFFNESS_7520_14,
            },
            damping={
                "waist": DAMPING_7520_14,
            },
            armature={
                "waist": ARMATURE_7520_14,
            },
        ),
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_pelvis_1", ".*_pelvis_2", ".*_thigh", ".*_calf"],
            effort_limit_sim=AK80_64["peak_torque"],
            velocity_limit_sim=AK80_64["rated_speed"],
            stiffness={
                ".*_pelvis_1": STIFFNESS_7520_14,
                ".*_pelvis_2": STIFFNESS_7520_22,
                ".*_thigh": STIFFNESS_7520_14,
                ".*_calf": STIFFNESS_7520_22,
            },
            damping={
                ".*_pelvis_1": DAMPING_7520_14,
                ".*_pelvis_2": DAMPING_7520_22,
                ".*_thigh": DAMPING_7520_14,
                ".*_calf": DAMPING_7520_22,
            },
            armature={
                ".*_pelvis_1": ARMATURE_7520_14,
                ".*_pelvis_2": ARMATURE_7520_22,
                ".*_thigh": ARMATURE_7520_14,
                ".*_calf": ARMATURE_7520_22,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_1", ".*_ankle_2"],
            effort_limit_sim=AK70_10["peak_torque"],
            velocity_limit_sim=AK70_10["rated_speed"],
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        "wheel": ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel"],
            effort_limit_sim=AK80_64["peak_torque"],
            velocity_limit_sim=AK80_64["rated_speed"],
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
        "foot_wheel": ImplicitActuatorCfg(
            joint_names_expr=[".*_foot_wheel_.*"],
            effort_limit_sim=0.0, # no actuation
            velocity_limit_sim=1000.0, # no limit
            stiffness=0.0,
            damping=0.1,
            armature=0.0,
            friction=0.0,
        ),
    }
)

WALKING_MODE_ACTION_SCALE = {}
for a in WHEEL_LEG_HUMANOID_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            WALKING_MODE_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
