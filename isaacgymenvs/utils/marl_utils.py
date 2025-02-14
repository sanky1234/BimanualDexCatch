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
