

from rl_games_twk.envs.connect4_network import ConnectBuilder
from rl_games_twk.envs.test_network import TestNetBuilder
from rl_games_twk.algos_torch import model_builder

model_builder.register_network('connect4net', ConnectBuilder)
model_builder.register_network('testnet', TestNetBuilder)