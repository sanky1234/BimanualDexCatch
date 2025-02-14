from abc import ABC

from .vec_task import Env
from typing import Dict, Any, Tuple, List, Set


class MultiVecTask(Env, ABC):
    def __init__(self, config: Dict[str, Any], rl_device: str, sim_device: str, graphics_device_id: int,
                 headless: bool):
        print("This is MultiVecTask Parent class......!")
        super().__init__(config, rl_device, sim_device, graphics_device_id, headless)

