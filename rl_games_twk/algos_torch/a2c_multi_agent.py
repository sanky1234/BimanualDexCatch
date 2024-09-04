import copy

from rl_games_twk.common.a2c_common import print_statistics, swap_and_flatten01
from rl_games_twk.common.experience import ExperienceBuffer
from rl_games_twk.algos_torch import torch_ext
from rl_games_twk.common import datasets
from rl_games_twk.common import common_losses

from torch import optim
from torch import nn

import torch
import torch.distributed as dist

import os
import time
from gym import spaces

import numpy as np

from .a2c_continuous import A2CAgent
from ..common.a2c_common import A2CBase


def split_list(input_list, num_agent):
    chunk_size = len(input_list) // num_agent
    return [input_list[i * chunk_size:(i + 1) * chunk_size] for i in range(num_agent)]


class OverridableDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def override(self, **kwargs):
        for key, value in kwargs.items():
            if key in self:
                self[key] = value
        return self


class MultiAgentA2CAgent(A2CAgent):
    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        self.num_multi_agents = self.vec_env.env.num_multi_agents
        self.num_a_actions = self.vec_env.env.num_a_actions     # another agent's action dim.

        self.dataset_list = [datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn,
                                                 self.ppo_device, self.seq_length) for _ in range(self.num_multi_agents)]

        self.current_rewards_list = []
        self.current_shaped_rewards_list = []

        self.game_rewards_list = []
        self.game_shaped_rewards_list = []

        self.last_lr_list = [self.last_lr for _ in range(self.num_multi_agents)]

        batch_size = self.num_agents * self.num_actors
        current_rewards_shape = (batch_size, self.value_size)
        for agent_id in range(self.num_multi_agents):
            curr_rew1 = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.ppo_device)
            curr_rew2 = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.ppo_device)
            self.current_rewards_list.append(curr_rew1)
            self.current_shaped_rewards_list.append(curr_rew2)

            avg_meter = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
            avg_meter_shaped = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
            self.game_rewards_list.append(avg_meter)
            self.game_shaped_rewards_list.append(avg_meter_shaped)

        # for Model
        catch_build_config = OverridableDict({'actions_num': self.actions_num - self.num_a_actions,
                                             'input_shape': self.obs_shape['catch'],
                                             'num_seqs': self.num_actors * self.num_agents,
                                             'value_size': self.env_info.get('value_size', 1),
                                             'normalize_value': self.normalize_value,
                                             'normalize_input': self.normalize_input})
        throw_build_config = copy.deepcopy(catch_build_config)
        throw_build_config['input_shape'] = self.obs_shape['throw']
        self.build_configs = [catch_build_config,
                              throw_build_config.override(actions_num=self.num_a_actions)]

        # for Experience Buffer
        self.env_info_list = [copy.deepcopy(self.env_info) for _ in range(len(self.build_configs))]
        self.env_info_list[0]['observation_space'] = self.env_info['observation_space']['catch']
        self.env_info_list[1]['observation_space'] = self.env_info['observation_space']['throw']

        for env_info, config in zip(self.env_info_list, self.build_configs):
            actions_num = config['actions_num']
            env_info['action_space'] = spaces.Box(np.ones(actions_num) * -1., np.ones(actions_num) * 1.)

        algo_info = {
            'num_actors': self.num_actors,
            'horizon_length': self.horizon_length,
            'has_central_value': self.has_central_value,
            'use_action_masks': self.use_action_masks
        }

        # generate rest of the models
        self.models = []
        self.optimizers = []
        self.exp_buffs = []
        if self.num_multi_agents > 1:
            # root model, exp_buffs
            for agent_id in range(self.num_multi_agents):
                # build other models
                model, optim = self.build_model(self.build_configs[agent_id], agent_id)
                self.models.append(model)
                self.optimizers.append(optim)

                # create other experience buffers
                self.exp_buffs.append(ExperienceBuffer(self.env_info_list[agent_id], algo_info, self.ppo_device))

        # Remove the unnecessary model and experience buffer that were created by the parent class.
        del self.model
        del self.optimizer

        if self.normalize_value:
            if not self.has_central_value:
                self.value_mean_std_list = []
                for agent_id in range(self.num_multi_agents):
                    self.value_mean_std_list.append(self.models[agent_id].value_mean_std)

    def build_model(self, build_config, agent_id):
        model = self.network.build(build_config)
        model.to(self.ppo_device)
        self.init_rnn_from_model(model)
        optimizer = optim.Adam(model.parameters(), float(self.last_lr_list[agent_id]), eps=1e-08, weight_decay=self.weight_decay)
        return model, optimizer

    def obs_to_tensors(self, obs):
        obs_is_dict = isinstance(obs, dict)
        if obs_is_dict:
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

    def env_reset(self):
        # observations
        obs_bufs = self.vec_env.env.obs_bufs
        for agent_id in range(self.num_multi_agents):
            obs_bufs["obs"+str(agent_id)] = torch.clamp(obs_bufs["obs"+str(agent_id)], -self.vec_env.env.clip_obs, self.vec_env.env.clip_obs).to(self.vec_env.env.rl_device)
        # obs_bufs = torch.clamp(obs_bufs, -self.vec_env.env.clip_obs, self.vec_env.env.clip_obs).to(self.vec_env.env.rl_device)
        obs = self.vec_env.reset()
        obs = self.obs_to_tensors(obs)

        # states if it has
        state_buf = obs['states'] if hasattr(obs, 'states') else obs_bufs

        sub_agent_obs = []
        agent_state = []

        for i in range(self.num_multi_agents):
            key = 'obs' + str(i)
            sub_agent_obs.append(obs[key])
            agent_state.append(state_buf[key])

        # to tensor form
        # _obs = torch.transpose(torch.stack(sub_agent_obs), 1, 0)
        # _state_all = torch.transpose(torch.stack(agent_state), 1, 0)

        return obs, agent_state

    def env_step(self, actions):
        actions = self.preprocess_actions(actions)
        obs, rewards, dones, infos = self.vec_env.step(actions)

        if self.is_tensor_obses:
            # value_size condition is removed..
            return self.obs_to_tensors(obs), rewards.to(self.ppo_device), dones.to(self.ppo_device), infos
        else:
            return self.obs_to_tensors(obs), torch.from_numpy(rewards).to(self.ppo_device).float(), torch.from_numpy(dones).to(self.ppo_device), infos

    def get_action_values(self, agent_id, obs):
        processed_obs = self._preproc_obs(obs['obs' + str(agent_id)])
        self.models[agent_id].eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs' : processed_obs,
            'rnn_states' : self.rnn_states
        }

        with torch.no_grad():
            res_dict = self.models[agent_id](input_dict)
            if self.has_central_value:
                states = obs['states']
                input_dict = {
                    'is_train': False,
                    'states' : states,
                }
                value = self.get_central_value(input_dict)
                res_dict['values'] = value
        return res_dict

    def write_stats(self, total_time, epoch_num, step_time, play_time, update_time, a_losses, c_losses, entropies, kls, last_lr_list, lr_mul, frame, scaled_time, scaled_play_time, curr_frames):
        # do we need scaled time?
        self.diagnostics.send_info(self.writer)
        self.writer.add_scalar('performance/step_inference_rl_update_fps', curr_frames / scaled_time, frame)
        self.writer.add_scalar('performance/step_inference_fps', curr_frames / scaled_play_time, frame)
        self.writer.add_scalar('performance/step_fps', curr_frames / step_time, frame)
        self.writer.add_scalar('performance/rl_update_time', update_time, frame)
        self.writer.add_scalar('performance/step_inference_time', play_time, frame)
        self.writer.add_scalar('performance/step_time', step_time, frame)
        for agent_id in range(self.num_multi_agents):
            self.writer.add_scalar('losses/a_loss'+str(agent_id), torch_ext.mean_list(a_losses[agent_id]).item(), frame)
            self.writer.add_scalar('losses/c_loss'+str(agent_id), torch_ext.mean_list(c_losses[agent_id]).item(), frame)

            self.writer.add_scalar('losses/entropy'+str(agent_id), torch_ext.mean_list(entropies[agent_id]).item(), frame)
            self.writer.add_scalar('info/kl'+str(agent_id), torch_ext.mean_list(kls[agent_id]).item(), frame)

            self.writer.add_scalar('info/last_lr'+str(agent_id), last_lr_list[agent_id] * lr_mul, frame)

        self.writer.add_scalar('info/lr_mul', lr_mul, frame)
        self.writer.add_scalar('info/e_clip', self.e_clip * lr_mul, frame)
        self.writer.add_scalar('info/epochs', epoch_num, frame)
        self.algo_observer.after_print_stats(frame, epoch_num, total_time)

    def set_eval(self):
        [self.models[agent_id].eval() for agent_id in range(self.num_multi_agents)]
        if self.normalize_rms_advantage:
            self.advantage_mean_std.eval()

    def set_train(self):
        [self.models[agent_id].train() for agent_id in range(self.num_multi_agents)]
        if self.normalize_rms_advantage:
            self.advantage_mean_std.train()

    def update_lr(self, lr, agent_id):
        if self.multi_gpu:
            lr_tensor = torch.tensor([lr], device=self.device)
            dist.broadcast(lr_tensor, 0)
            lr = lr_tensor.item()

        for param_group in self.optimizers[agent_id].param_groups:
            param_group['lr'] = lr

        # if self.has_central_value:
        #    self.central_value_net.update_lr(lr)

    def get_values(self, obs):
        values = []
        for agent_id in range(self.num_multi_agents):
            with torch.no_grad():
                if self.has_central_value:
                    states = obs['states']
                    self.central_value_net.eval()
                    input_dict = {
                        'is_train': False,
                        'states': states,
                        'actions': None,
                        'is_done': self.dones,
                    }
                    value = self.get_central_value(input_dict)
                else:
                    self.models[agent_id].eval()
                    processed_obs = self._preproc_obs(obs['obs' + str(agent_id)])
                    input_dict = {
                        'is_train': False,
                        'prev_actions': None,
                        'obs': processed_obs,
                        'rnn_states': self.rnn_states
                    }
                    result = self.models[agent_id](input_dict)
                    value = result['values']
                    values.append(value)
        return values

    def prepare_dataset(self, batch_dict_list):
        for agent_id in range(self.num_multi_agents):
            obses = batch_dict_list[agent_id]['obses']
            returns = batch_dict_list[agent_id]['returns']
            dones = batch_dict_list[agent_id]['dones']
            values = batch_dict_list[agent_id]['values']
            actions = batch_dict_list[agent_id]['actions']
            neglogpacs = batch_dict_list[agent_id]['neglogpacs']
            mus = batch_dict_list[agent_id]['mus']
            sigmas = batch_dict_list[agent_id]['sigmas']
            rnn_states = batch_dict_list[agent_id].get('rnn_states', None)
            rnn_masks = batch_dict_list[agent_id].get('rnn_masks', None)

            advantages = returns - values

            if self.normalize_value:    # this should be updated for multi-agent...
                self.value_mean_std_list[agent_id].train()
                values = self.value_mean_std_list[agent_id](values)
                returns = self.value_mean_std_list[agent_id](returns)
                self.value_mean_std_list[agent_id].eval()

            advantages = torch.sum(advantages, axis=1)

            if self.normalize_advantage:
                if self.is_rnn:
                    if self.normalize_rms_advantage:
                        advantages = self.advantage_mean_std(advantages, mask=rnn_masks)
                    else:
                        advantages = torch_ext.normalization_with_masks(advantages, rnn_masks)
                else:
                    if self.normalize_rms_advantage:
                        advantages = self.advantage_mean_std(advantages)
                    else:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            dataset_dict = {}
            dataset_dict['old_values'] = values
            dataset_dict['old_logp_actions'] = neglogpacs
            dataset_dict['advantages'] = advantages
            dataset_dict['returns'] = returns
            dataset_dict['actions'] = actions
            dataset_dict['obs'] = obses
            dataset_dict['dones'] = dones
            dataset_dict['rnn_states'] = rnn_states
            dataset_dict['rnn_masks'] = rnn_masks
            dataset_dict['mu'] = mus
            dataset_dict['sigma'] = sigmas

            self.dataset_list[agent_id].update_values_dict(dataset_dict)

            if self.has_central_value:
                dataset_dict = {}
                dataset_dict['old_values'] = values
                dataset_dict['advantages'] = advantages
                dataset_dict['returns'] = returns
                dataset_dict['actions'] = actions
                dataset_dict['obs'] = batch_dict_list[agent_id]['states']
                dataset_dict['dones'] = dones
                dataset_dict['rnn_masks'] = rnn_masks
                self.central_value_net.update_dataset(dataset_dict)     # should be updated for multi-agent setting

    def get_stats_weights(self, model_stats=False):
        state = {}
        if self.mixed_precision:
            state['scaler'] = self.scaler.state_dict()
        if self.has_central_value:
            state['central_val_stats'] = self.central_value_net.get_stats_weights(model_stats)
        if model_stats:
            if self.normalize_input:
                for agent_id in range(self.num_multi_agents):
                    state['running_mean_std'+str(agent_id)] = self.models[agent_id].running_mean_std.state_dict()
            if self.normalize_value:
                for agent_id in range(self.num_multi_agents):
                    state['reward_mean_std'+str(agent_id)] = self.models[agent_id].value_mean_std.state_dict()

        return state

    def get_weights(self):
        state = self.get_stats_weights()
        for agent_id in range(self.num_multi_agents):
            state['model'+str(agent_id)] = self.models[agent_id].state_dict()
        return state

    def get_full_state_weights(self):
        state = self.get_weights()
        state['epoch'] = self.epoch_num
        state['frame'] = self.frame
        for agent_id in range(self.num_multi_agents):
            state['optimizer'+str(agent_id)] = self.optimizers[agent_id].state_dict()

        if self.has_central_value:
            state['assymetric_vf_nets'] = self.central_value_net.state_dict()

        # This is actually the best reward ever achieved. last_mean_rewards is perhaps not the best variable name
        # We save it to the checkpoint to prevent overriding the "best ever" checkpoint upon experiment restart
        state['last_mean_rewards'] = self.last_mean_rewards

        if self.vec_env is not None:
            env_state = self.vec_env.get_env_state()
            state['env_state'] = env_state

        return state

    def play_steps(self):
        update_list = self.update_list

        step_time = 0.0

        for n in range(self.horizon_length):
            res_list = []
            for agent_id in range(self.num_multi_agents):
                if self.use_action_masks:
                    masks = self.vec_env.get_action_masks()
                    res_dict = self.get_masked_action_values(self.obs, masks)
                else:
                    res_dict = self.get_action_values(agent_id, self.obs)   # here
                self.exp_buffs[agent_id].update_data('obses', n, self.obs['obs' + str(agent_id)])
                self.exp_buffs[agent_id].update_data('dones', n, self.dones)

                res_list.append(res_dict)
                for k in update_list:
                    self.exp_buffs[agent_id].update_data(k, n, res_list[agent_id][k])
                if self.has_central_value:
                    self.exp_buffs[agent_id].update_data('states', n, self.obs['states'])

            # Unified one-step action
            unified_actions = torch.cat([item['actions'] for item in res_list], dim=-1)
            step_time_start = time.time()
            self.obs, rewards, self.dones, infos = self.env_step(unified_actions)
            step_time_end = time.time()

            step_time += (step_time_end - step_time_start)

            shaped_rewards = self.rewards_shaper(rewards)
            if self.value_bootstrap and 'time_outs' in infos:
                for agent_id in range(self.num_multi_agents):
                    discounted_reward = (self.gamma * res_list[agent_id]['values'] *
                                         self.cast_obs(infos['time_outs']).unsqueeze(1).float())
                    shaped_rewards[:, agent_id] += discounted_reward.squeeze(1)
                    self.exp_buffs[agent_id].update_data('rewards', n, shaped_rewards[:, agent_id].unsqueeze(-1))

            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices[::self.num_agents]

            _curr_rewards = []
            _curr_shaped_rewards = []
            for agent_id in range(self.num_multi_agents):
                self.current_rewards_list[agent_id] += rewards[:, agent_id].unsqueeze(-1)
                self.current_shaped_rewards_list[agent_id] += shaped_rewards[:, agent_id].unsqueeze(-1)

                self.game_rewards_list[agent_id].update(self.current_rewards_list[agent_id][env_done_indices])
                self.game_shaped_rewards_list[agent_id].update(self.current_shaped_rewards_list[agent_id][env_done_indices])
                _curr_rewards.append(self.current_rewards_list[agent_id][env_done_indices])
                _curr_shaped_rewards.append(self.current_shaped_rewards_list[agent_id][env_done_indices])

            self.game_rewards.update(sum(_curr_rewards))
            self.game_shaped_rewards.update(sum(_curr_shaped_rewards))
            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.dones.float()

            for agent_id in range(self.num_multi_agents):
                self.current_rewards_list[agent_id] = self.current_rewards_list[agent_id] * not_dones.unsqueeze(1)
                self.current_shaped_rewards_list[agent_id] = self.current_shaped_rewards_list[agent_id] * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        last_values = self.get_values(self.obs)

        batch_dict_list = []
        fdones = self.dones.float()
        for agent_id in range(self.num_multi_agents):
            mb_fdones = self.exp_buffs[agent_id].tensor_dict['dones'].float()
            mb_values = self.exp_buffs[agent_id].tensor_dict['values']
            mb_rewards = self.exp_buffs[agent_id].tensor_dict['rewards']
            mb_advs = self.discount_values(fdones, last_values[agent_id], mb_fdones, mb_values, mb_rewards)
            mb_returns = mb_advs + mb_values

            batch_dict = self.exp_buffs[agent_id].get_transformed_list(swap_and_flatten01, self.tensor_list)
            batch_dict['returns'] = swap_and_flatten01(mb_returns)
            batch_dict['played_frames'] = self.batch_size
            batch_dict['step_time'] = step_time
            batch_dict_list.append(batch_dict)

        return batch_dict_list

    def play_steps_rnn(self):
        update_list = self.update_list
        mb_rnn_states = self.mb_rnn_states
        step_time = 0.0

        for n in range(self.horizon_length):
            if n % self.seq_length == 0:
                for s, mb_s in zip(self.rnn_states, mb_rnn_states):
                    mb_s[n // self.seq_length, :, :, :] = s

            if self.has_central_value:
                self.central_value_net.pre_step_rnn(n)

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)

            self.rnn_states = res_dict['rnn_states']
            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones.byte())

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])
            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            step_time_start = time.time()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            step_time_end = time.time()

            step_time += (step_time_end - step_time_start)

            shaped_rewards = self.rewards_shaper(rewards)

            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(
                    1).float()

            self.experience_buffer.update_data('rewards', n, shaped_rewards)

            self.current_rewards += rewards
            self.current_shaped_rewards += shaped_rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices[::self.num_agents]

            if len(all_done_indices) > 0:
                if self.zero_rnn_on_done:
                    for s in self.rnn_states:
                        s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0
                if self.has_central_value:
                    self.central_value_net.post_step_rnn(all_done_indices)

            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_shaped_rewards.update(self.current_shaped_rewards[env_done_indices])
            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_shaped_rewards = self.current_shaped_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        last_values = self.get_values(self.obs)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()

        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values
        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)

        batch_dict['returns'] = swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        states = []
        for mb_s in mb_rnn_states:
            t_size = mb_s.size()[0] * mb_s.size()[2]
            h_size = mb_s.size()[3]
            states.append(mb_s.permute(1, 2, 0, 3).reshape(-1, t_size, h_size))

        batch_dict['rnn_states'] = states
        batch_dict['step_time'] = step_time

        return batch_dict

    def train_epoch(self):
        A2CBase.train_epoch(self)   # Grandparent class

        self.set_eval()
        play_time_start = time.time()

        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict_list = self.play_steps()

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks_list = []
        for agent_id in range(self.num_multi_agents):
            rnn_masks = batch_dict_list[agent_id].get('rnn_masks', None)
            rnn_masks_list.append(rnn_masks)

        self.set_train()
        # current_frames for all agents?
        self.curr_frames = np.mean([batch_dict_list[agent_id].pop('played_frames')
                                    for agent_id in range(self.num_multi_agents)], dtype=int)
        self.prepare_dataset(batch_dict_list)
        self.algo_observer.after_steps()
        if self.has_central_value:
            self.train_central_value()

        a_losses = []
        c_losses = []
        b_losses = []
        entropies = []
        kls = []

        for mini_ep in range(0, self.mini_epochs_num):
            for agent_id in torch.randperm(self.num_multi_agents):  # randomized agent id, TODO
                ep_kls = []     # per agent
                for i in range(len(self.dataset_list[agent_id])):
                    a_loss, c_loss, entropy, kl, last_lr_list, lr_mul, cmu, csigma, b_loss = self.train_actor_critic(self.dataset_list[agent_id][i], agent_id)
                    a_losses.append(a_loss)
                    c_losses.append(c_loss)
                    ep_kls.append(kl)
                    entropies.append(entropy)
                    if self.bounds_loss_coef is not None:
                        b_losses.append(b_loss)

                    self.dataset_list[agent_id].update_mu_sigma(cmu, csigma)
                    if self.schedule_type == 'legacy':
                        av_kls = kl
                        if self.multi_gpu:
                            dist.all_reduce(kl, op=dist.ReduceOp.SUM)
                            av_kls /= self.world_size
                        self.last_lr_list[agent_id], self.entropy_coef = self.scheduler.update(self.last_lr_list[agent_id],
                                                                                               self.entropy_coef, self.epoch_num, 0, av_kls.item())
                        self.update_lr(self.last_lr_list[agent_id], agent_id)

                av_kls = torch_ext.mean_list(ep_kls)
                if self.multi_gpu:
                    dist.all_reduce(av_kls, op=dist.ReduceOp.SUM)
                    av_kls /= self.world_size
                if self.schedule_type == 'standard':    # default!
                    self.last_lr_list[agent_id], self.entropy_coef = self.scheduler.update(self.last_lr_list[agent_id],
                                                                                           self.entropy_coef, self.epoch_num, 0, av_kls.item())
                    self.update_lr(self.last_lr_list[agent_id], agent_id)

                kls.append(av_kls)
                self.diagnostics.mini_epoch(self, mini_ep)
                if self.normalize_input:
                    self.models[agent_id].running_mean_std.eval()  # don't need to update statstics more than one miniepoch

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        # reshape the list
        a_losses = split_list(a_losses, self.num_multi_agents)
        c_losses = split_list(c_losses, self.num_multi_agents)
        b_losses = split_list(b_losses, self.num_multi_agents)
        entropies = split_list(entropies, self.num_multi_agents)
        kls = split_list(kls, self.num_multi_agents)

        return (batch_dict_list[-1]['step_time'], play_time, update_time, total_time,
                a_losses, c_losses, b_losses, entropies, kls, last_lr_list, lr_mul)

    def train(self):
        self.init_tensors()
        self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        self.obs, _ = self.env_reset()  # self.obs, share_obs = self.env_reset()
        self.curr_frames = self.batch_size_envs

        if self.multi_gpu:
            print("====================broadcasting parameters")
            model_params = [self.model.state_dict()]
            dist.broadcast_object_list(model_params, 0)
            self.model.load_state_dict(model_params[0])

        while True:
            epoch_num = self.update_epoch()
            step_time, play_time, update_time, sum_time, a_losses, c_losses, b_losses, entropies, kls, last_lr_list, lr_mul = self.train_epoch()
            total_time += sum_time
            frame = self.frame // self.num_agents

            # cleaning memory to optimize space
            for agent_id in range(self.num_multi_agents):
                self.dataset_list[agent_id].update_values_dict(None)
            should_exit = False

            if self.global_rank == 0:
                self.diagnostics.epoch(self, current_epoch=epoch_num)
                # do we need scaled_time?
                scaled_time = self.num_agents * sum_time
                scaled_play_time = self.num_agents * play_time
                curr_frames = self.curr_frames * self.world_size if self.multi_gpu else self.curr_frames
                self.frame += curr_frames

                print_statistics(self.print_stats, curr_frames, step_time, scaled_play_time, scaled_time,
                                 epoch_num, self.max_epochs, frame, self.max_frames)

                self.write_stats(total_time, epoch_num, step_time, play_time, update_time,
                                 a_losses, c_losses, entropies, kls, last_lr_list, lr_mul, frame,
                                 scaled_time, scaled_play_time, curr_frames)

                if len(b_losses) > 0:
                    for agent_id in range(self.num_multi_agents):
                        self.writer.add_scalar('losses/bounds_loss'+str(agent_id),
                                               torch_ext.mean_list(b_losses[agent_id]).item(), frame)

                if self.has_soft_aug:
                    self.writer.add_scalar('losses/aug_loss', np.mean(aug_losses), frame)

                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_shaped_rewards = self.game_shaped_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()
                    self.mean_rewards = mean_rewards[0]

                    for i in range(self.value_size):    # default: value_size=1
                        rewards_name = 'rewards' if i == 0 else 'rewards{0}'.format(i)
                        self.writer.add_scalar(rewards_name + '/step'.format(i), mean_rewards[i], frame)
                        self.writer.add_scalar(rewards_name + '/iter'.format(i), mean_rewards[i], epoch_num)
                        self.writer.add_scalar(rewards_name + '/time'.format(i), mean_rewards[i], total_time)
                        self.writer.add_scalar('shaped_' + rewards_name + '/step'.format(i), mean_shaped_rewards[i], frame)
                        self.writer.add_scalar('shaped_' + rewards_name + '/iter'.format(i), mean_shaped_rewards[i], epoch_num)
                        self.writer.add_scalar('shaped_' + rewards_name + '/time'.format(i), mean_shaped_rewards[i], total_time)

                    self.writer.add_scalar('episode_lengths/step', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)
                    self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)

                    if self.has_self_play_config:
                        self.self_play_manager.update(self)

                    checkpoint_name = self.config['name'] + '_ep_' + str(epoch_num) + '_rew_' + str(mean_rewards[0])

                    if self.save_freq > 0:
                        if epoch_num % self.save_freq == 0:
                            self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))

                    if mean_rewards[0] > self.last_mean_rewards and epoch_num >= self.save_best_after:
                        print('saving next best rewards: ', mean_rewards)
                        self.last_mean_rewards = mean_rewards[0]
                        self.save(os.path.join(self.nn_dir, self.config['name']))

                        if 'score_to_win' in self.config:
                            if self.last_mean_rewards > self.config['score_to_win']:
                                print('Maximum reward achieved. Network won!')
                                self.save(os.path.join(self.nn_dir, checkpoint_name))
                                should_exit = True

                if epoch_num >= self.max_epochs and self.max_epochs != -1:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max epochs reached before any env terminated at least once')
                        mean_rewards = -np.inf

                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_ep_' + str(epoch_num) \
                                           + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                    print('MAX EPOCHS NUM!')
                    should_exit = True

                if self.frame >= self.max_frames and self.max_frames != -1:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max frames reached before any env terminated at least once')
                        mean_rewards = -np.inf

                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_frame_' + str(self.frame) \
                                           + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                    print('MAX FRAMES NUM!')
                    should_exit = True

                update_time = 0

            if self.multi_gpu:
                should_exit_t = torch.tensor(should_exit, device=self.device).float()
                dist.broadcast(should_exit_t, 0)
                should_exit = should_exit_t.float().item()
            if should_exit:
                return self.last_mean_rewards, epoch_num

            if should_exit:
                return self.last_mean_rewards, epoch_num

    def train_actor_critic(self, input_dict, agent_id):
        self.calc_gradients(input_dict, agent_id)
        return self.train_result

    def calc_gradients(self, input_dict, agent_id):
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)

        lr_mul = 1.0
        curr_e_clip = self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch,
            'obs': obs_batch,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_length

            if self.zero_rnn_on_done:
                batch_dict['dones'] = input_dict['dones']

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.models[agent_id](batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip)

            if self.has_value_loss:
                c_loss = common_losses.critic_loss(self.models[agent_id], value_preds_batch, values, curr_e_clip, return_batch,
                                                   self.clip_value)
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)
            if self.bound_loss_type == 'regularisation':
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == 'bound':
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)
            losses, sum_mask = torch_ext.apply_masks(
                [a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

            loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.models[agent_id].parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        # TODO: Refactor this ugliest code of they year
        self.trancate_gradients_and_step(agent_id)

        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  # / sum_mask

        self.diagnostics.mini_batch(self,
                                    {
                                        'values': value_preds_batch,
                                        'returns': return_batch,
                                        'new_neglogp': action_log_probs,
                                        'old_neglogp': old_action_log_probs_batch,
                                        'masks': rnn_masks
                                    }, curr_e_clip, 0)

        self.train_result = (a_loss, c_loss, entropy, kl_dist, self.last_lr_list, lr_mul, mu.detach(), sigma.detach(), b_loss)

    def trancate_gradients_and_step(self, agent_id):
        if self.multi_gpu:
            # batch allreduce ops: see https://github.com/entity-neural-network/incubator/pull/220
            all_grads_list = []
            for param in self.models[agent_id].parameters():
                if param.grad is not None:
                    all_grads_list.append(param.grad.view(-1))

            all_grads = torch.cat(all_grads_list)
            dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
            offset = 0
            for param in self.models[agent_id].parameters():
                if param.grad is not None:
                    param.grad.data.copy_(
                        all_grads[offset : offset + param.numel()].view_as(param.grad.data) / self.world_size
                    )
                    offset += param.numel()

        if self.truncate_grads:
            self.scaler.unscale_(self.optimizers[agent_id])
            nn.utils.clip_grad_norm_(self.models[agent_id].parameters(), self.grad_norm)

        self.scaler.step(self.optimizers[agent_id])
        self.scaler.update()
