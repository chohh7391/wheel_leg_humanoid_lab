import argparse
from isaaclab.app import AppLauncher

# 1. Argparse 설정
parser = argparse.ArgumentParser(description="Tutorial on inferencing a policy on an H1 robot in a warehouse.")
parser.add_argument("--mode", type=str, help="Select locomotion mode (walking or driving)", required=True)
parser.add_argument("--rough", action="store_true", default=False, help="Select rough terrain")
parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint exported as jit.", required=True)
# [수정] type=str 제거, action="store_true"로 변경 (플래그로 사용)
parser.add_argument("--plot", action="store_true", default=False, help="Plot applied torque data")

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

PLOT_JOINT_NAMES = [
    "waist_yaw_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]

def main():
    # load the trained jit policy
    policy_path = os.path.abspath(args_cli.checkpoint)
    # omni.client.read_file 이슈 방지를 위해 직접 로드
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

    # 속도 명령 고정 (테스트용)
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.8, 0.8)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    
    # create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    robot = env.scene["robot"]
    _, joint_names = robot.find_joints(".*")

    # ---------------------------------------------------------
    # Plot 초기화 (args_cli.plot이 True일 때만 실행)
    # ---------------------------------------------------------
    plot_indices = []
    lines = {}
    data_history = {}
    history_len = 100 # deque 길이

    if args_cli.plot:
        plot_indices = [joint_names.index(name) for name in PLOT_JOINT_NAMES if name in joint_names]
        num_plots = len(plot_indices)
        
        ncols = 3 
        nrows = int(np.ceil(num_plots / ncols))
        
        plt.ion()
        fig, ax_flat = plt.subplots(nrows, ncols, figsize=(18, 3.5 * nrows), sharex=True)
        ax = ax_flat.flatten() if nrows * ncols > 1 else [ax_flat]

        fig.suptitle('Real-time Applied Motor Torque Monitoring (AK80-64)', fontsize=16, weight='bold')
        
        # deque로 초기화 (성능 최적화)
        data_history = {name: collections.deque(maxlen=history_len) for name in PLOT_JOINT_NAMES}
        
        for i in range(num_plots):
            name = PLOT_JOINT_NAMES[i]
            
            # 스타일 설정
            ax[i].set_title(f"{name}", fontsize=12, loc='left', color='dodgerblue')
            ax[i].grid(True, linestyle='--', alpha=0.4)
            ax[i].set_ylabel("Torque (Nm)", fontsize=10)
            
            # [수정] Y축 범위 고정 (-130 ~ 130)
            ax[i].set_ylim(-130, 130)
            ax[i].set_xlim(0, history_len) # X축 초기 범위 설정
            
            # 기준선 그리기
            ax[i].axhline(y=RATED_TORQUE, color='orange', linestyle='--', linewidth=1.0, alpha=0.7, label='Rated (48)')
            ax[i].axhline(y=-RATED_TORQUE, color='orange', linestyle='--', linewidth=1.0, alpha=0.7)
            ax[i].axhline(y=PEAK_TORQUE, color='red', linestyle='-', linewidth=1.2, alpha=0.6, label='Peak (120)')
            ax[i].axhline(y=-PEAK_TORQUE, color='red', linestyle='-', linewidth=1.2, alpha=0.6)

            # 데이터 라인
            line, = ax[i].plot([], [], label='Applied', color='lime', linewidth=1.5)
            lines[name] = line
            
            # 범례
            handles, labels = ax[i].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax[i].legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8, framealpha=0.8)

            if i >= num_plots:
                ax[i].set_visible(False)
                
        plt.tight_layout(rect=[0, 0, 1, 0.97])

    # ---------------------------------------------------------
    # Inference Loop
    # ---------------------------------------------------------
    obs, _ = env.reset()
    step_count = 0
    PLOT_INTERVAL = 15 # 15 스텝마다 렌더링 (최적화)

    with torch.inference_mode():
        while simulation_app.is_running():
            action = policy(obs["policy"])
            obs, _, _, _, _ = env.step(action)
            
            # Plot이 켜져 있을 때만 데이터 처리
            if args_cli.plot:
                applied_torque = robot.data.applied_torque.squeeze()
                
                if applied_torque.dim() == 1 and len(applied_torque) == len(joint_names):
                    # 1. 데이터 수집 (매 스텝 수행)
                    for name, idx in zip(PLOT_JOINT_NAMES, plot_indices):
                        torque_val = applied_torque[idx].item()
                        data_history[name].append(torque_val)

                    # 2. 그래프 그리기 (N 스텝마다 수행하여 랙 방지)
                    if step_count % PLOT_INTERVAL == 0:
                        for name in PLOT_JOINT_NAMES:
                            if name in lines: # 키 존재 확인
                                current_data = data_history[name]
                                if len(current_data) > 0:
                                    lines[name].set_data(range(len(current_data)), current_data)
                        
                        fig.canvas.draw()
                        fig.canvas.flush_events()
                
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