import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tutorial on inferencing a policy on an H1 robot in a warehouse.")
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
import matplotlib.pyplot as plt
import numpy as np


PLOT_JOINT_NAMES = [
    "waist",
    "right_pelvis_1", "right_pelvis_2", "left_pelvis_1", "left_pelvis_2",
    "right_thigh", "left_thigh",
    "right_calf", "left_calf",
    "right_ankle_1", "right_ankle_2", "left_ankle_1", "left_ankle_2",
]


def main():
    # load the trained jit policy
    policy_path = os.path.abspath(args_cli.checkpoint)
    file_content = omni.client.read_file(policy_path)[2]
    file = io.BytesIO(memoryview(file_content).tobytes())
    policy = torch.jit.load(file, map_location=args_cli.device)

    # setup environment
    env_cfg = WheelLegHumanoidWalkingRoughEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.curriculum = None
    env_cfg.terminations.terrain_out_of_bounds = None
    env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Grid/default_environment.usd",
    )
    env_cfg.sim.device = args_cli.device
    if args_cli.device == "cpu":
        env_cfg.sim.use_fabric = False

    # create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    robot = env.scene["robot"]
    joint_ids, joint_names = robot.find_joints(".*")

    plot_indices = [joint_names.index(name) for name in PLOT_JOINT_NAMES if name in joint_names]
    
    num_plots = len(plot_indices)
    
    # 서브플롯 구조: 4열로 나누어 가로로 확장
    ncols = 3 
    nrows = int(np.ceil(num_plots / ncols))
    
    # 창 크기를 더 크고 넓게 설정
    plt.ion()
    # fig.size: 가로 18인치, 세로 (행 수에 비례)
    fig, ax_flat = plt.subplots(nrows, ncols, figsize=(18, 3.5 * nrows), sharex=True)
    
    # ax_flat을 1차원 리스트로 변환하여 인덱싱을 쉽게 함
    ax = ax_flat.flatten() if nrows * ncols > 1 else [ax_flat]

    fig.suptitle('Real-time Applied Motor Torque Monitoring (AK80-64)', fontsize=16, weight='bold')
    
    history_len = 500
    data_history = {name: [] for name in PLOT_JOINT_NAMES}
    lines = {}
    
    # 플롯 생성 및 스타일 설정
    for i in range(num_plots):
        name = PLOT_JOINT_NAMES[i]
        
        # 1. 제목 및 그리드 설정
        ax[i].set_title(f"{name}", fontsize=12, loc='left', color='dodgerblue')
        ax[i].grid(True, linestyle='--', alpha=0.6)
        ax[i].set_ylabel("Torque (Nm)", fontsize=10)
        
        # 2. 라인 스타일 설정
        line, = ax[i].plot([], [], label=name, color='lime', linewidth=1.5)
        lines[name] = line
        
        # 3. 플롯에 사용되지 않는 빈 공간 처리 (선택 사항)
        if i >= num_plots:
            ax[i].set_visible(False)
            
    # X축 라벨은 가장 아래 행에만 표시
    for i in range(num_plots - ncols, num_plots):
        if i < num_plots:
            ax[i].set_xlabel("Time Steps (Simulation Frames)", fontsize=10)
    
    # 플롯 간 간격 조정 (plt.tight_layout을 사용하면 서브플롯들이 겹치지 않게 배치됨)
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
                        
                        # Y축 자동 조정 및 최소 범위 보장
                        if current_data:
                            y_min = min(current_data)
                            y_max = max(current_data)
                            y_range = y_max - y_min
                            
                            # 데이터가 거의 변화 없을 때를 대비해 최소 20Nm 범위 확보
                            if y_range < 20.0:
                                center = (y_min + y_max) / 2
                                y_min = center - 10.0
                                y_max = center + 10.0
                            else:
                                # 데이터가 변화할 경우, 5Nm 마진 추가
                                y_min -= 5.0
                                y_max += 5.0
                                
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