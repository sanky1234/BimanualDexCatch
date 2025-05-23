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
    football_pose = torch.zeros(13, device=envs.device)
    football_pose[0] = -0.2
    football_pose[2] = 0.8
    football_pose[6] = 1.0
    envs.set_initial_football_state(football_pose)

    num_elites = int(envs.num_envs * 0.25)

    mean = torch.zeros(len(envs.joint_idx_mapping), device=envs.device)
    std = torch.ones(len(envs.joint_idx_mapping), device=envs.device) * 1.0
    low = envs._dof_lower_limits[envs.joint_idx_mapping]
    high = envs._dof_upper_limits[envs.joint_idx_mapping]
    for step in range(1000):
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
        
        elites = actions[elite_indices]
        
        mean = elites.mean(axis=0)
        std = elites.std(axis=0)

        if verbose:
            print(f"Iter {step}: best reward = {rewards[elite_indices[-1]]:.3f}, mean = {mean}, std = {std}")


if __name__ == "__main__":
    launch_cem()