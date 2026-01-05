import argparse
from isaaclab.app import AppLauncher

# 1. Argparse 설정
parser = argparse.ArgumentParser(description="Tutorial on inferencing a policy on an H1 robot in a warehouse.")
parser.add_argument("--mode", type=str, help="Select locomotion mode (walking or driving)", required=True)
parser.add_argument("--rough", action="store_true", default=False, help="Select rough terrain")
parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint exported as jit.", required=True)
parser.add_argument("--plot", action="store_true", default=False, help="Plot applied torque and velocity data")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import io
import os
import torch
import omni
import collections # deque 사용을 위해 추가

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from wheel_leg_humanoid_lab.tasks.manager_based.wheel_leg_humanoid.velocity.config.walking.rough_env_cfg import WheelLegHumanoidWalkingRoughEnvCfg
from wheel_leg_humanoid_lab.tasks.manager_based.wheel_leg_humanoid.velocity.config.walking.flat_env_cfg import WheelLegHumanoidWalkingFlatEnvCfg
from wheel_leg_humanoid_lab.tasks.manager_based.wheel_leg_humanoid.velocity.config.driving.rough_env_cfg import WheelLegHumanoidDrivingRoughEnvCfg
from wheel_leg_humanoid_lab.tasks.manager_based.wheel_leg_humanoid.velocity.config.driving.flat_env_cfg import WheelLegHumanoidDrivingFlatEnvCfg

import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------
# 모터 스펙 상수 정의
# ------------------------------------------------------------------
PEAK_TORQUE = 120.0
RATED_TORQUE = 48.0
PEAK_SPEED = 5.654866776461628
RATED_SPEED = 5.026548245743669

PLOT_JOINT_NAMES = [
    "waist_yaw_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]

def main():
    # load the trained jit policy
    policy_path = os.path.abspath(args_cli.checkpoint)
    policy = torch.jit.load(policy_path, map_location=args_cli.device)

    # setup environment
    if args_cli.mode == "walking":
        if args_cli.rough:
            env_cfg = WheelLegHumanoidWalkingRoughEnvCfg()
        else:
            env_cfg = WheelLegHumanoidWalkingFlatEnvCfg()
    elif args_cli.mode == "driving":
        if args_cli.rough:
            env_cfg = WheelLegHumanoidDrivingRoughEnvCfg()
        else:
            env_cfg = WheelLegHumanoidDrivingFlatEnvCfg()
    else:
        raise ValueError("--mode must be 'walking' or 'driving'")
    
    if not args_cli.rough:
        env_cfg.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="usd",
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Grid/default_environment.usd",
        )
    
    env_cfg.scene.num_envs = 1
    env_cfg.curriculum = None
    env_cfg.terminations.terrain_out_of_bounds = None
    
    env_cfg.sim.device = args_cli.device
    if args_cli.device == "cpu":
        env_cfg.sim.use_fabric = False

    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.8, 0.8)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    
    env = ManagerBasedRLEnv(cfg=env_cfg)
    robot = env.scene["robot"]
    _, joint_names = robot.find_joints(".*")

    # ---------------------------------------------------------
    # Plot 초기화 (창 2개 생성: Torque, Velocity)
    # ---------------------------------------------------------
    plot_indices = []
    lines_torque = {}
    lines_vel = {}
    data_history_torque = {}
    data_history_vel = {}
    history_len = 100 

    if args_cli.plot:
        plot_indices = [joint_names.index(name) for name in PLOT_JOINT_NAMES if name in joint_names]
        num_plots = len(plot_indices)
        ncols = 3 
        nrows = int(np.ceil(num_plots / ncols))
        
        plt.ion()
        
        # [창 1] Torque Plot
        fig_torque, ax_t_flat = plt.subplots(nrows, ncols, figsize=(15, 3 * nrows), sharex=True)
        ax_t = ax_t_flat.flatten()
        fig_torque.suptitle('Applied Torque Monitoring (Nm)', fontsize=14, weight='bold')

        # [창 2] Velocity Plot
        fig_vel, ax_v_flat = plt.subplots(nrows, ncols, figsize=(15, 3 * nrows), sharex=True)
        ax_v = ax_v_flat.flatten()
        fig_vel.suptitle('Joint Velocity Monitoring (rad/s)', fontsize=14, weight='bold', color='darkred')

        data_history_torque = {name: collections.deque(maxlen=history_len) for name in PLOT_JOINT_NAMES}
        data_history_vel = {name: collections.deque(maxlen=history_len) for name in PLOT_JOINT_NAMES}
        
        for i in range(num_plots):
            name = PLOT_JOINT_NAMES[i]
            idx = plot_indices[i]
            
            # --- Torque 설정 ---
            ax_t[i].set_title(f"{name}", fontsize=10, loc='left', color='dodgerblue')
            ax_t[i].set_ylim(-130, 130)
            ax_t[i].axhline(y=RATED_TORQUE, color='orange', linestyle='--', linewidth=1.0, alpha=0.7, label='Rated (48)')
            ax_t[i].axhline(y=-RATED_TORQUE, color='orange', linestyle='--', linewidth=1.0, alpha=0.7)
            ax_t[i].axhline(y=PEAK_TORQUE, color='red', linestyle='-', linewidth=1.2, alpha=0.6, label='Peak (120)')
            ax_t[i].axhline(y=-PEAK_TORQUE, color='red', linestyle='-', linewidth=1.2, alpha=0.6)
            line_t, = ax_t[i].plot([], [], color='lime', linewidth=1.2)
            lines_torque[name] = line_t

            # --- Velocity 설정 ---
            ax_v[i].set_title(f"{name}", fontsize=10, loc='left', color='darkred')
            ax_v[i].set_ylim(-PEAK_SPEED * 1.5, PEAK_SPEED * 1.5) # 리미트의 1.5배로 범위 설정
            ax_v[i].axhline(y=RATED_SPEED, color='orange', linestyle='--', linewidth=1.0, alpha=0.7, label='Rated (48)')
            ax_v[i].axhline(y=-RATED_SPEED, color='orange', linestyle='--', linewidth=1.0, alpha=0.7)
            ax_v[i].axhline(y=PEAK_SPEED, color='red', linestyle='-', linewidth=1.2, alpha=0.6, label='Peak (120)')
            ax_v[i].axhline(y=-PEAK_SPEED, color='red', linestyle='-', linewidth=1.2, alpha=0.6)
            line_v, = ax_v[i].plot([], [], color='blue', linewidth=1.2, label='Current')
            lines_vel[name] = line_v

        fig_torque.tight_layout(rect=[0, 0, 1, 0.97])
        fig_vel.tight_layout(rect=[0, 0, 1, 0.97])

    # ---------------------------------------------------------
    # Inference Loop
    # ---------------------------------------------------------
    obs, _ = env.reset()
    step_count = 0
    PLOT_INTERVAL = 15 

    with torch.inference_mode():
        while simulation_app.is_running():
            action = policy(obs["policy"])
            obs, _, _, _, _ = env.step(action)
            
            if args_cli.plot:
                applied_torque = robot.data.applied_torque.squeeze()
                joint_vel = robot.data.joint_vel.squeeze()
                
                # 1. 데이터 수집
                for name, idx in zip(PLOT_JOINT_NAMES, plot_indices):
                    data_history_torque[name].append(applied_torque[idx].item())
                    data_history_vel[name].append(joint_vel[idx].item())

                # 2. 그래프 업데이트 (15스텝마다)
                if step_count % PLOT_INTERVAL == 0:
                    for name in PLOT_JOINT_NAMES:
                        # Torque 업데이트
                        lines_torque[name].set_data(range(len(data_history_torque[name])), data_history_torque[name])
                        ax_t[PLOT_JOINT_NAMES.index(name)].set_xlim(0, history_len)
                        
                        # Velocity 업데이트
                        lines_vel[name].set_data(range(len(data_history_vel[name])), data_history_vel[name])
                        ax_v[PLOT_JOINT_NAMES.index(name)].set_xlim(0, history_len)
                    
                    fig_torque.canvas.draw()
                    fig_torque.canvas.flush_events()
                    fig_vel.canvas.draw()
                    fig_vel.canvas.flush_events()
                
            step_count += 1

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        simulation_app.close()
        plt.ioff()
        print("\n--- 시뮬레이션 종료 ---")
        plt.show()