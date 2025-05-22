import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.1", config_name="config_opt", config_path="./cfg")
def launch_cem(cfg: DictConfig): 
    """
    Launches the CEM Optimization Process with given configrations. 
    """
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
    
    for step in range(1000):
        actions = torch.zeros((envs.num_envs, len(envs.joint_idx_mapping)), device=envs.device)  # no-op actions
        actions[:, 0] = ((0.1 * step) % 3.14)


        envs.step(actions)
        # import pdb; pdb.set_trace()




if __name__ == "__main__":
    launch_cem()