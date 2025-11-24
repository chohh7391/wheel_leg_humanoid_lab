import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
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


def cal_motor_armature(inertia, reduction):
    """
    <input dim>
    inertia: g cm^2
    reduction: ?:1
    """
    # convert unit: g cm^2 -> kg m^2
    inertia = inertia * 1e-3 * (1e-2)**2
    armature = inertia * reduction**2
    return armature


AK80_64 = {
    "peak_torque": 120,
    "rated_torque": 48,
    "rated_speed": rpm2rad_per_s(48),
    "inertia": 564.5,  # g cm^2
    "reduction": 64,  # 64:1
}

NATURAL_FREQ = 10 * 2.0 * np.pi  # 10Hz
DAMPING_RATIO = 2.0

ARMATURE_AK80_64 = cal_motor_armature(AK80_64["inertia"], AK80_64["reduction"])
STIFFNESS_AK80_64 = ARMATURE_AK80_64 * NATURAL_FREQ**2
DAMPING_AK80_64 = 2.0 * DAMPING_RATIO * ARMATURE_AK80_64 * NATURAL_FREQ


# WHEEL_RADIUS = 0.11
# L = 0.54

# MAX_LINVEL = WHEEL_RADIUS * AK80_64["rated_speed"]
# MAX_ANGVEL = MAX_LINVEL / L


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
        pos=(0.0, 0.0, 0.78),
        joint_pos={
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim=AK80_64["peak_torque"],
            velocity_limit_sim=AK80_64["rated_speed"],
            stiffness=STIFFNESS_AK80_64,
            damping=DAMPING_AK80_64,
            armature=ARMATURE_AK80_64,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=AK80_64["peak_torque"],
            velocity_limit_sim=AK80_64["rated_speed"],
            stiffness=2.0 * STIFFNESS_AK80_64,
            damping=2.0 * DAMPING_AK80_64,
            armature=2.0 * ARMATURE_AK80_64,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim=AK80_64["peak_torque"],
            velocity_limit_sim=AK80_64["rated_speed"],
            stiffness=STIFFNESS_AK80_64,
            damping=DAMPING_AK80_64,
            armature=ARMATURE_AK80_64,
        ),
        "wheel": ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            effort_limit_sim=AK80_64["peak_torque"],
            velocity_limit_sim=AK80_64["rated_speed"],
            stiffness=0.0,
            damping=0.5,
            armature=ARMATURE_AK80_64,
        ),
        "foot_wheel": ImplicitActuatorCfg(
            joint_names_expr=[".*_foot_wheel_joint_.*"],
            effort_limit_sim=0.0,  # no actuation
            velocity_limit_sim=1000.0,  # no limit
            stiffness=0.0,
            damping=0.5,
            armature=0.0,
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
