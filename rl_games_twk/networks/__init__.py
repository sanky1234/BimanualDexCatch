from rl_games_twk.networks.tcnn_mlp import TcnnNetBuilder
from rl_games_twk.algos_torch import model_builder

model_builder.register_network('tcnnnet', TcnnNetBuilder)