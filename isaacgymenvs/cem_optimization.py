import hydra
from omegaconf import DictConfig, OmegaConf
import logging 
import os 
from datetime import datetime 
    
import isaacgym 
from hydra.utils import to_absolute_path 
from isaacgymenvs.tasks import isaacgym_task_map 
import gym 
from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict 
from isaacgymenvs.utils.utils import set_np_formatting, set_seed 
import isaacgymenvs
import torch
from torchrl.modules import TruncatedNormal
import numpy as np
from scipy.spatial.transform import Rotation as R
import itertools
import pandas as pd
from collections import deque

def truncated_gaussian_samples(mean, std, low, high, num_samples):
    """
    Generate samples from a truncated Gaussian distribution.
    """

    mu = mean.unsqueeze(0).expand(num_samples, -1)
    scale = std.unsqueeze(0).expand(num_samples, -1)
    low_limit = low.unsqueeze(0).expand(num_samples, -1)
    high_limit = high.unsqueeze(0).expand(num_samples, -1)

    dist = TruncatedNormal(loc=mu, scale=scale, low=low_limit, high=high_limit)
    samples = dist.rsample()
    return samples

@hydra.main(version_base="1.1", config_name="config_opt", config_path="./cfg")
def launch_cem(cfg: DictConfig): 
    """
    Launches the CEM Optimization Process with given configrations. 
    """
    

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # set numpy formatting for printing 
    set_np_formatting()

    envs = isaacgymenvs.make(
        cfg.seed, 
        cfg.task_name, 
        cfg.task.env.numEnvs, 
        cfg.sim_device,
        cfg.rl_device,
        cfg.graphics_device_id,
        cfg.headless,
        cfg.multi_gpu,
        cfg.capture_video,
        cfg.force_render,
        cfg
    )
    if cfg.capture_video:
        pass
        # envs.is_vector_env = True
        # envs = gym.wrappers.RecordVideo(
        #     envs,
        #     f"videos/{run_name}",
        #     step_trigger=lambda step: step % cfg.capture_video_freq == 0,
        #     video_length=cfg.capture_video_len,
        # )

    verbose = True
    percent_elite = 0.25
    max_steps = 100
    # Ranges for position and orientation
    x_range = np.arange(0, 0.487 + 1e-6, 0.1)
    y_range = np.arange(0.0, 0.1 + 1e-6, 0.1)
    z_range = np.arange(0.8, 1.52 + 1e-6, 0.1)
    pitch_range = np.arange(0, np.pi/6.0 + 1e-6, np.pi/3.0)
    yaw_range = np.arange(0, np.pi/6.0 + 1e-6, np.pi/3.0)
    # roll_range = np.arange(-np.pi/6.0, np.pi/6.0 + 1e-6, np.pi/6.0)

    stagnation_window = 10
    stagnation_thresh = 1e-3
    buffer_size = 0.5 * percent_elite

    for x_pose in x_range:
        for y_pose in y_range:
            for z_pose in z_range:
                for ball_pitch in pitch_range:
                    for ball_yaw in yaw_range:
                        # for ball_roll in roll_range:
                            football_pose = torch.zeros(13, device=envs.device)
                            football_pose[0] = x_pose
                            football_pose[1] = y_pose
                            football_pose[2] = z_pose
                            ball_roll = 0
                            r = R.from_euler('xyz', [ball_pitch, ball_yaw, ball_roll], degrees=False)
                            football_quat = r.as_quat() # returns (x, y, z, w)
                            football_pose[3] = football_quat[0]
                            football_pose[4] = football_quat[1]
                            football_pose[5] = football_quat[2]
                            football_pose[6] = football_quat[3]


                            football_vel = [5.0, 0.0, 0.0]  # initial ball linear velocity
                            football_vel_adjusted = r.apply(football_vel)  # adjust velocity according to orientation

                            football_pose[7] = football_vel_adjusted[0]  # ball linear velocity x
                            football_pose[8] = football_vel_adjusted[1] # ball linear velocity y
                            football_pose[9] = football_vel_adjusted[2] # ball linear velocity z
                            football_pose[10] = 0.0
                            football_pose[11] = 0.0
                            football_pose[12] = 0.0


                            envs.set_initial_football_state(football_pose)
                            num_elites = int(envs.num_envs * percent_elite)
                            buffer_len = int(num_elites * buffer_size)
                            mean = torch.zeros(len(envs.joint_idx_mapping), device=envs.device)
                            std = torch.ones(len(envs.joint_idx_mapping), device=envs.device) * 1.0
                            low = envs._dof_lower_limits[envs.joint_idx_mapping]
                            high = envs._dof_upper_limits[envs.joint_idx_mapping]
                            std_history = deque(maxlen=stagnation_window)
                            # elite_buffer = torch.zeros((buffer_len, len(envs.joint_idx_mapping)), device=envs.device)
                            elite_buffer = []
                            for step in range(max_steps):
                                    actions = truncated_gaussian_samples(
                                        mean=mean,
                                        std=std,
                                        low=low,
                                        high=high,
                                        num_samples=envs.num_envs
                                    )
                                    # actions = torch.zeros((envs.num_envs, len(envs.joint_idx_mapping)), device=envs.device)  # no-op actions
                                    # actions[:, 0] = ((0.1 * step) % 3.14)


                                    envs.step(actions)
                                    # import pdb; pdb.set_trace()

                                    rewards = envs.total_reward
                                    elite_indices = rewards.argsort()[-num_elites:]  # top-k
                                    
                                    elites = envs.actual_joint_states[elite_indices]

                                    positive_elites = torch.where(elites > 0)[0]
                                    if len(positive_elites) > 0:
                                        pos_elite_rewards = rewards[positive_elites]
                                        pos_elite_actions = envs.actual_joint_states[positive_elites]

                                        for i in range(len(pos_elite_rewards)):
                                            elite_buffer.append((pos_elite_actions[i].clone(), pos_elite_rewards[i].item()))

                                        elite_buffer = sorted(elite_buffer, key=lambda x: x[1], reverse=True)[:buffer_len]
                                        buffer_elites = torch.stack([x[0] for x in elite_buffer], dim=0)

                                        elites = torch.cat((elites, buffer_elites), dim=0)  # combine elites with buffer
                    
                                    
                                    mean = elites.mean(axis=0)
                                    std = elites.std(axis=0)

                                    std_history.append(std.clone())
                                    if len(std_history) == stagnation_window:
                                        std_diff = torch.abs(std_history[-1] - std_history[0])
                                        if torch.all(std_diff < stagnation_thresh):
                                            print(f"Early stopping at iteration {iteration} due to variance stagnation.")
                                            break

                                    if verbose:
                                        print(f"Iter {step}: best reward = {rewards[elite_indices[-1]]:.3f}, mean = {mean}, std = {std}")
                            elites_np = elites.cpu().numpy()  # (N, action_dim)

                            rows = [[x_pose, y_pose, z_pose, ball_pitch, ball_yaw, ball_roll] + list(elite) for elite in elites_np]

                            pose_headers = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
                            action_headers = [name for name in envs.input_control_names]
                            headers = pose_headers + action_headers
                            df = pd.DataFrame(rows, columns=headers)
                            current_date = datetime.now().strftime("%Y-%m-%d")



                            df.to_csv(f"./cem_results/May31/football_pose_{x_pose}_{y_pose}_{z_pose}_{ball_pitch}_{ball_yaw}_{ball_roll}_{current_date}.csv", index=False)
                            print(f"Saved elite actions for football pose {x_pose}, {y_pose}, {z_pose}, {ball_pitch}, {ball_yaw}, {ball_roll} to CSV.")


if __name__ == "__main__":
    launch_cem()