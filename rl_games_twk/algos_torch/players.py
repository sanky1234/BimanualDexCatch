from rl_games_twk.common.player import BasePlayer
from rl_games_twk.algos_torch import torch_ext
from rl_games_twk.algos_torch.running_mean_std import RunningMeanStd
from rl_games_twk.common.tr_helpers import unsqueeze_obs
import gym
import torch 
from torch import nn
import numpy as np
import copy

import os
import csv

from .a2c_multi_agent import OverridableDict


def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action = action * d + m
    return scaled_action

class MultiAgentPpoPlayerContinuous(BasePlayer):

    def __init__(self, params):
        BasePlayer.__init__(self, params)
        self.network = self.config['network']
        self.actions_num = self.action_space.shape[0]
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        self.mask = [False]

        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)

        self.num_multi_agents = self.env.num_multi_agents
        self.num_a_actions = self.env.num_a_actions

        obs_shape = self.obs_shape
        catch_build_config = OverridableDict({
            'actions_num' : self.actions_num - self.num_a_actions,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        })
        throw_build_config = copy.deepcopy(catch_build_config)

        self.build_configs = [catch_build_config,
                              throw_build_config.override(actions_num=self.num_a_actions)]
        self.build_configs[0]['input_shape'] = self.build_configs[0]['input_shape']['catch']
        self.build_configs[1]['input_shape'] = self.build_configs[1]['input_shape']['throw']

        assert len(self.build_configs) == self.num_multi_agents

        self.models = []
        for agent_id in range(self.num_multi_agents):
            model = self.network.build(self.build_configs[agent_id])
            model.to(self.device)
            model.eval()
            self.is_rnn = model.is_rnn()   # ?
            self.models.append(model)

    def get_batch_size(self, obses, batch_size):
        obs_shape = self.obs_shape[next(iter(self.obs_shape))]

        if isinstance(obs_shape, dict):
            obses = obses.get('obs', obses)
            first_key = 'observation' if 'observation' in obses else next(iter(obs_shape))
            obs_shape = obs_shape[first_key]
            obses = obses[first_key]

        if isinstance(obses, dict):
            first_key = next(iter(obses))
            obs = obses[first_key]
        else:
            obs = obses

        if len(obs.size()) > len(obs_shape):
            batch_size = obs.size(0)
            self.has_batch_dimension = True

        self.batch_size = batch_size
        return batch_size

    def get_action(self, obs, is_deterministic = False):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)

        mus = []
        actions = []
        for agent_id in range(self.num_multi_agents):
            input_dict = {
                'is_train': False,
                'prev_actions': None,
                'obs': obs['obs'+str(agent_id)],
                'rnn_states': self.states
            }
            with torch.no_grad():
                res_dict = self.models[agent_id](input_dict)
            mu = res_dict['mus']
            action = res_dict['actions']

            mus.append(mu)
            actions.append(action)
            self.states = res_dict['rnn_states']    # None

        mus = torch.cat(mus, dim=-1).to(self.device)
        actions = torch.cat(actions, dim=-1).to(self.device)

        if is_deterministic:
            current_action = mus
        else:
            current_action = actions
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())

        if self.clip_actions:
            return rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        for agent_id in range(self.num_multi_agents):
            self.models[agent_id].load_state_dict(checkpoint['model'+str(agent_id)])
            if self.normalize_input and 'running_mean_std' in checkpoint:
                self.models[agent_id].running_mean_std.load_state_dict(checkpoint['running_mean_std'+str(agent_id)])

        env_state = checkpoint.get('env_state', None)
        if self.env is not None and env_state is not None:
            self.env.set_env_state(env_state)

    def save_rew_to_csv(self, path, total_rew_list):
        # save total_rew_list as .csv format with the name including total_n_games,
        if all(isinstance(item, (float, int)) for item in total_rew_list):
            total_rew_list = [[item] for item in total_rew_list]  # Convert to list of lists

        base_path = os.path.dirname(os.path.dirname(path))
        eval_folder_path = os.path.join(base_path, "eval")

        if not os.path.exists(eval_folder_path):
            os.makedirs(eval_folder_path)

        # Define the csv file path (with noise scale value, epinum)
        ns_val = str(self.env.noise_scale)  # ex) ns1.5 --> noise_scale=1.5
        games_num = str(self.games_num)
        file_name = (os.path.splitext(os.path.basename(base_path))[0] +
                     '_ns' + ns_val + '_epi' + games_num + '.csv')
        csv_file_path = os.path.join(eval_folder_path, file_name)

        # Write the list data to the csv file
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(total_rew_list)

    def reset(self):
        self.init_rnn()

    def run(self, args=None):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_deterministic = self.is_deterministic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True
            # print('setting agent weights for selfplay')
            # self.env.create_agent(self.env.config)
            # self.env.set_weights(range(8),self.get_weights())

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        self.wait_for_checkpoint()

        total_rew_list = []

        need_init_rnn = self.is_rnn
        for _ in range(n_games):
            if games_played >= n_games:
                break

            obses = self.env_reset(self.env)
            batch_size = 1
            batch_size = self.get_batch_size(obses, batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32)
            steps = torch.zeros(batch_size, dtype=torch.float32)

            print_game_res = False

            for n in range(self.max_steps):
                if self.evaluation and n % self.update_checkpoint_freq == 0:
                    self.maybe_load_new_checkpoint()

                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(
                        obses, masks, is_deterministic)
                else:
                    action = self.get_action(obses, is_deterministic)

                obses, r, done, info = self.env_step(self.env, action)
                cr += torch.sum(r, dim=-1)
                steps += 1

                if render:
                    self.env.render(mode='human')
                    time.sleep(self.render_sleep)

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)
                games_played += done_count

                if done_count > 0:
                    if self.is_rnn:
                        for s in self.states:
                            s[:, all_done_indices, :] = s[:,
                                                          all_done_indices, :] * 0.0

                    total_rew_list += cr[done_indices].flatten().tolist()
                    cur_rewards = cr[done_indices].sum().item()
                    cur_steps = steps[done_indices].sum().item()

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())
                    sum_rewards += cur_rewards
                    sum_steps += cur_steps

                    game_res = 0.0
                    if isinstance(info, dict):
                        if 'battle_won' in info:
                            print_game_res = True
                            game_res = info.get('battle_won', 0.5)
                        if 'scores' in info:
                            print_game_res = True
                            game_res = info.get('scores', 0.5)

                    if self.print_stats:
                        cur_rewards_done = cur_rewards/done_count
                        cur_steps_done = cur_steps/done_count
                        if print_game_res:
                            print(f'reward: {cur_rewards_done:.2f} steps: {cur_steps_done:.1f} w: {game_res}')
                        else:
                            print(f'reward: {cur_rewards_done:.2f} steps: {cur_steps_done:.1f}')

                    sum_game_res += game_res
                    if batch_size//self.num_agents == 1 or games_played >= n_games:
                        break

        print(sum_rewards)
        if print_game_res:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps /
                  games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
        else:
            print('av reward:', sum_rewards / games_played * n_game_life,
                  'av steps:', sum_steps / games_played * n_game_life)
            self.save_rew_to_csv(path=args["checkpoint"], total_rew_list=total_rew_list)


class PpoPlayerContinuous(BasePlayer):

    def __init__(self, params):
        BasePlayer.__init__(self, params)
        self.network = self.config['network']
        self.actions_num = self.action_space.shape[0] 
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        self.mask = [False]

        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)

        obs_shape = self.obs_shape
        config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        } 
        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()

    def get_action(self, obs, is_deterministic = False):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : obs,
            'rnn_states' : self.states
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict['mus']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if is_deterministic:
            current_action = mu
        else:
            current_action = action
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())

        if self.clip_actions:
            return rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

        env_state = checkpoint.get('env_state', None)
        if self.env is not None and env_state is not None:
            self.env.set_env_state(env_state)

    def reset(self):
        self.init_rnn()


class PpoPlayerDiscrete(BasePlayer):

    def __init__(self, params):
        BasePlayer.__init__(self, params)

        self.network = self.config['network']
        if type(self.action_space) is gym.spaces.Discrete:
            self.actions_num = self.action_space.n
            self.is_multi_discrete = False
        if type(self.action_space) is gym.spaces.Tuple:
            self.actions_num = [action.n for action in self.action_space]
            self.is_multi_discrete = True
        self.mask = [False]
        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)
        obs_shape = self.obs_shape
        config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        }

        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()

    def get_masked_action(self, obs, action_masks, is_deterministic = True):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        action_masks = torch.Tensor(action_masks).to(self.device).bool()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : obs,
            'action_masks' : action_masks,
            'rnn_states' : self.states
        }
        self.model.eval()

        with torch.no_grad():
            res_dict = self.model(input_dict)
        logits = res_dict['logits']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if self.is_multi_discrete:
            if is_deterministic:
                action = [torch.argmax(logit.detach(), axis=-1).squeeze() for logit in logits]
                return torch.stack(action,dim=-1)
            else:    
                return action.squeeze().detach()
        else:
            if is_deterministic:
                return torch.argmax(logits.detach(), axis=-1).squeeze()
            else:    
                return action.squeeze().detach()

    def get_action(self, obs, is_deterministic = False):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)

        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : obs,
            'rnn_states' : self.states
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        logits = res_dict['logits']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if self.is_multi_discrete:
            if is_deterministic:
                action = [torch.argmax(logit.detach(), axis=1).squeeze() for logit in logits]
                return torch.stack(action,dim=-1)
            else:    
                return action.squeeze().detach()
        else:
            if is_deterministic:
                return torch.argmax(logits.detach(), axis=-1).squeeze()
            else:    
                return action.squeeze().detach()

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

        env_state = checkpoint.get('env_state', None)
        if self.env is not None and env_state is not None:
            self.env.set_env_state(env_state)

    def reset(self):
        self.init_rnn()


class SACPlayer(BasePlayer):

    def __init__(self, params):
        BasePlayer.__init__(self, params)
        self.network = self.config['network']
        self.actions_num = self.action_space.shape[0] 
        self.action_range = [
            float(self.env_info['action_space'].low.min()),
            float(self.env_info['action_space'].high.max())
        ]

        obs_shape = self.obs_shape
        self.normalize_input = False
        config = {
            'obs_dim': self.env_info["observation_space"].shape[0],
            'action_dim': self.env_info["action_space"].shape[0],
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'value_size': self.env_info.get('value_size', 1),
            'normalize_value': False,
            'normalize_input': self.normalize_input,
        }  
        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.sac_network.actor.load_state_dict(checkpoint['actor'])
        self.model.sac_network.critic.load_state_dict(checkpoint['critic'])
        self.model.sac_network.critic_target.load_state_dict(checkpoint['critic_target'])
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

        env_state = checkpoint.get('env_state', None)
        if self.env is not None and env_state is not None:
            self.env.set_env_state(env_state)

    def get_action(self, obs, is_deterministic=False):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        dist = self.model.actor(obs)
        actions = dist.sample() if is_deterministic else dist.mean
        actions = actions.clamp(*self.action_range).to(self.device)
        if self.has_batch_dimension == False:
            actions = torch.squeeze(actions.detach())
        return actions

    def reset(self):
        pass
