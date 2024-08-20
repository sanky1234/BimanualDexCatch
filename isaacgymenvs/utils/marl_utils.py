import numpy as np
import torch
from gym import spaces

from .rlgames_utils import RLGPUEnv


class MultiAgentRLGPUEnv(RLGPUEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        super().__init__(config_name, num_actors, **kwargs)

        # temp variables for MARL
        self.num_agents = self.env.num_agents
        self.num_share_observations = self.env.num_observations
        self.share_observations_space = [spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.num_share_observations, ))
                                         for _ in range(self.num_agents)]

    def reset(self):
        for agent_id in range(self.env.num_multi_agents):
            self.env.obs_dict["obs" + str(agent_id)] = (
                torch.clamp(self.env.obs_buf, -self.env.clip_obs, self.env.clip_obs).to(self.env.rl_device))
            # TODO, obses should be different for each agent..

        # asymmetric actor-critic
        if self.env.num_states > 0:
            self.env.obs_dict["states"] = self.env.get_state()

        return self.env.obs_dict

    def step(self, actions):
        obs_dict, rew_buf, reset_buf, extras = super().step(actions)
        if self.env.num_multi_agents > 1:
            for agent_id in range(self.env.num_multi_agents):
                obs_dict["obs" + str(agent_id)] = (
                    torch.clamp(self.env.obs_buf, -self.env.clip_obs, self.env.clip_obs).to(self.env.rl_device))
            # TODO, obses should be different for each agent..
            del obs_dict["obs"]

        return obs_dict, rew_buf, reset_buf, extras