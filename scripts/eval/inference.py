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


PLOT_JOINT_NAMES = ["waist", "right_pelvis_1", "left_pelvis_1"]


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
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
    )
    env_cfg.sim.device = args_cli.device
    if args_cli.device == "cpu":
        env_cfg.sim.use_fabric = False

    # create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    robot = env.scene["robot"]

    joint_ids, joint_names = robot.find_joints(".*")
    for i in range(len(joint_ids)):
        print(f"joint_id: {joint_ids[i]}, joint_name: {joint_names[i]}")
        
    plot_indices = [joint_names.index(name) for name in PLOT_JOINT_NAMES if name in joint_names]

    plt.ion()
    fig, ax = plt.subplots(len(plot_indices), 1, figsize=(10, 8), sharex=True)
    if len(plot_indices) == 1:
        ax = [ax]

    history_len = 500  # 최근 500 스텝의 데이터만 표시
    data_history = {name: [] for name in PLOT_JOINT_NAMES}

    lines = {}
    for i, name in enumerate(PLOT_JOINT_NAMES):
        if name in joint_names:
            ax[i].set_title(f"Applied Torque: {name}")
            ax[i].grid(True)

            line, _ = ax[i].plot([], [], label=name)
            lines[name] = line

    # run inference with the policy
    obs, _ = env.reset()
    with torch.inference_mode():
        while simulation_app.is_running():
            action = policy(obs["policy"])
            obs, _, _, _, _ = env.step(action)
            applied_torque = robot.data.applied_torque.squeeze()
            
            # 토크 값 로깅 및 플롯 업데이트
            if applied_torque.dim() == 1 and len(applied_torque) == len(joint_names):
                
                # 데이터 업데이트
                for name, idx in zip(PLOT_JOINT_NAMES, plot_indices):
                    torque_val = applied_torque[idx].item()
                    data_history[name].append(torque_val)
                    
                    # history_len 유지
                    if len(data_history[name]) > history_len:
                        data_history[name] = data_history[name][1:]

                # 플롯 업데이트
                time_steps = range(len(data_history[PLOT_JOINT_NAMES[0]]))
                for i, name in enumerate(PLOT_JOINT_NAMES):
                    if name in joint_names:
                        lines[name].set_data(time_steps, data_history[name])
                        
                        # x축 및 y축 자동 조정
                        ax[i].set_xlim(max(0, step_count - history_len), step_count)
                        y_min = min(data_history[name]) - 5
                        y_max = max(data_history[name]) + 5
                        ax[i].set_ylim(y_min, y_max)
                        
                fig.canvas.draw()
                fig.canvas.flush_events()
                
            step_count += 1


if __name__ == "__main__":
    main()
    simulation_app.close()
