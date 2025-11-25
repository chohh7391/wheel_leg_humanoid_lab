import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tutorial on inferencing a policy on an H1 robot in a warehouse.")
parser.add_argument("--mode", type=str, help="Select locomotion mode (walking or driving)", required=True)
parser.add_argument("--rough", action="store_true", default=False, help="Select rough terrian")
parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint exported as jit.", required=True)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import io
import os
import torch

import omni

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from wheel_leg_humanoid_lab.tasks.manager_based.wheel_leg_humanoid.velocity.config.walking.rough_env_cfg import WheelLegHumanoidWalkingRoughEnvCfg
from wheel_leg_humanoid_lab.tasks.manager_based.wheel_leg_humanoid.velocity.config.walking.flat_env_cfg import WheelLegHumanoidWalkingFlatEnvCfg
from wheel_leg_humanoid_lab.tasks.manager_based.wheel_leg_humanoid.velocity.config.driving.rough_env_cfg import WheelLegHumanoidDrivingRoughEnvCfg
from wheel_leg_humanoid_lab.tasks.manager_based.wheel_leg_humanoid.velocity.config.walking.flat_env_cfg import WheelLegHumanoidDrivingFlatEnvCfg

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
    file_content = omni.client.read_file(policy_path)[2]
    file = io.BytesIO(memoryview(file_content).tobytes())
    policy = torch.jit.load(file, map_location=args_cli.device)

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

    # create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    robot = env.scene["robot"]
    joint_ids, joint_names = robot.find_joints(".*")

    plot_indices = [joint_names.index(name) for name in PLOT_JOINT_NAMES if name in joint_names]
    
    num_plots = len(plot_indices)
    
    # 서브플롯 구조: 3열로 나누어 가로로 확장
    ncols = 3 
    nrows = int(np.ceil(num_plots / ncols))
    
    # 창 크기 설정
    plt.ion()
    fig, ax_flat = plt.subplots(nrows, ncols, figsize=(18, 3.5 * nrows), sharex=True)
    
    ax = ax_flat.flatten() if nrows * ncols > 1 else [ax_flat]

    fig.suptitle('Real-time Applied Motor Torque Monitoring (AK80-64)', fontsize=16, weight='bold')
    
    history_len = 50 # 히스토리 길이를 조금 늘려 추이를 더 잘 보이게 함
    data_history = {name: [] for name in PLOT_JOINT_NAMES}
    lines = {}
    
    # 플롯 생성 및 스타일 설정
    for i in range(num_plots):
        name = PLOT_JOINT_NAMES[i]
        
        # 1. 제목 및 그리드 설정
        ax[i].set_title(f"{name}", fontsize=12, loc='left', color='dodgerblue')
        ax[i].grid(True, linestyle='--', alpha=0.4)
        ax[i].set_ylabel("Torque (Nm)", fontsize=10)
        
        # 2. 모터 스펙 기준선 그리기 (Rated & Peak)
        # Rated Torque (±48) - 주황색 점선
        ax[i].axhline(y=RATED_TORQUE, color='orange', linestyle='--', linewidth=1.0, alpha=0.7, label='Rated (48)')
        ax[i].axhline(y=-RATED_TORQUE, color='orange', linestyle='--', linewidth=1.0, alpha=0.7)
        
        # Peak Torque (±120) - 빨간색 실선
        ax[i].axhline(y=PEAK_TORQUE, color='red', linestyle='-', linewidth=1.2, alpha=0.6, label='Peak (120)')
        ax[i].axhline(y=-PEAK_TORQUE, color='red', linestyle='-', linewidth=1.2, alpha=0.6)

        # 3. 데이터 라인 스타일 설정
        line, = ax[i].plot([], [], label='Applied', color='lime', linewidth=1.5)
        lines[name] = line
        
        # 4. 범례 추가 (첫 번째 루프에서만 모든 라벨을 처리하거나, 중복 제거)
        # 중복된 라벨 제거를 위한 처리
        handles, labels = ax[i].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax[i].legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8, framealpha=0.8)

        # 5. 빈 플롯 숨기기
        if i >= num_plots:
            ax[i].set_visible(False)
            
    # X축 라벨은 가장 아래 행에만 표시
    for i in range(num_plots - ncols, num_plots):
        if i < num_plots:
            ax[i].set_xlabel("Time Steps", fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # run inference with the policy
    obs, _ = env.reset()
    step_count = 0
    with torch.inference_mode():
        while simulation_app.is_running():
            action = policy(obs["policy"])
            obs, _, _, _, _ = env.step(action)
            applied_torque = robot.data.applied_torque.squeeze()
            
            if applied_torque.dim() == 1 and len(applied_torque) == len(joint_names):
                
                # 데이터 수집 및 업데이트
                for name, idx in zip(PLOT_JOINT_NAMES, plot_indices):
                    torque_val = applied_torque[idx].item()
                    data_history[name].append(torque_val)
                    
                    if len(data_history[name]) > history_len:
                        data_history[name] = data_history[name][1:]

                # 플롯 시각화 업데이트
                for i, name in enumerate(PLOT_JOINT_NAMES):
                    if name in joint_names:
                        current_data = data_history[name]
                        
                        # X축 데이터 크기 맞춤
                        x_data = range(max(0, step_count + 1 - len(current_data)), step_count + 1)
                        lines[name].set_data(x_data, current_data)
                        
                        # X축 (시간 윈도우) 업데이트
                        ax[i].set_xlim(max(0, step_count - history_len), step_count)
                        
                        # Y축 자동 조정 (기준선 고려)
                        if current_data:
                            curr_min = min(current_data)
                            curr_max = max(current_data)
                            
                            # 기본 마진
                            y_min = curr_min - 5.0
                            y_max = curr_max + 5.0
                            
                            # 데이터 범위가 너무 작으면(예: 0 근처), Rated Torque가 보일 정도로 살짝 넓혀줄 수 있음 (선택 사항)
                            # 하지만 파형을 자세히 보는 게 더 중요하므로, 최소 범위만 보장하고
                            # 데이터가 튀면 자동으로 확장되도록 기존 로직 유지 + 보완
                            y_range = y_max - y_min
                            if y_range < 20.0:
                                center = (curr_min + curr_max) / 2
                                y_min = center - 15.0 # 최소 ±15 범위 확보
                                y_max = center + 15.0
                            
                            ax[i].set_ylim(y_min, y_max)
                            
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
        print("\n--- 시뮬레이션 종료. 플롯 창을 닫아주세요. ---")
        plt.show()