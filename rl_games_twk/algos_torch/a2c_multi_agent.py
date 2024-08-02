import copy

from rl_games_twk.common.a2c_common import print_statistics, swap_and_flatten01
from rl_games_twk.common.experience import ExperienceBuffer
from rl_games_twk.algos_torch import torch_ext

from torch import optim
import torch
import torch.distributed as dist

import os
import time
from gym import spaces

import numpy as np

from .a2c_continuous import A2CAgent
from ..common.a2c_common import A2CBase


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

        self.current_rewards_list = []
        self.current_shaped_rewards_list = []

        # del self.game_rewards
        self.game_rewards_list = []
        self.game_shaped_rewards_list = []
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
                                             'input_shape': self.obs_shape,
                                             'num_seqs': self.num_actors * self.num_agents,
                                             'value_size': self.env_info.get('value_size', 1),
                                             'normalize_value': self.normalize_value,
                                             'normalize_input': self.normalize_input})
        throw_build_config = copy.deepcopy(catch_build_config)
        self.build_configs = [catch_build_config,
                              throw_build_config.override(actions_num=self.num_a_actions)]

        # for Experience Buffer
        self.env_info_list = [copy.deepcopy(self.env_info) for _ in range(len(self.build_configs))]
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
        self.exp_buffs = []
        if self.num_multi_agents > 1:
            # root model, exp_buffs
            for agent_id in range(self.num_multi_agents):
                # build other models
                model, optim = self.build_model(self.build_configs[agent_id])
                self.models.append(model)

                # create other experience buffers
                self.exp_buffs.append(ExperienceBuffer(self.env_info_list[agent_id], algo_info, self.ppo_device))

        # Remove the unnecessary model and experience buffer that were created by the parent class.
        del self.model

    def build_model(self, build_config):
        model = self.network.build(build_config)
        model.to(self.ppo_device)
        self.init_rnn_from_model(model)
        optimizer = optim.Adam(model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)
        return model, optimizer

    def env_reset(self):
        # observations
        obs_buf = self.vec_env.env.obs_buf
        obs_buf = torch.clamp(obs_buf, -self.vec_env.env.clip_obs, self.vec_env.env.clip_obs).to(self.vec_env.env.rl_device)
        obs = self.vec_env.reset()
        obs['obs1'] = obs_buf
        obs = self.obs_to_tensors(obs)

        # states if it has
        state_buf = obs['states'] if hasattr(obs, 'states') else obs_buf

        sub_agent_obs = []
        agent_state = []

        for i in range(self.num_multi_agents):
            key = 'obs' + str(i) if i > 0 else 'obs'
            sub_agent_obs.append(obs[key])
            agent_state.append(state_buf)

        # to tensor form
        _obs = torch.transpose(torch.stack(sub_agent_obs), 1, 0)
        _state_all = torch.transpose(torch.stack(agent_state), 1, 0)

        return obs, _state_all

    def env_step(self, actions):
        actions = self.preprocess_actions(actions)
        obs, rewards, dones, infos = self.vec_env.step(actions)

        if self.is_tensor_obses:
            # value_size condition is removed..
            return self.obs_to_tensors(obs), rewards.to(self.ppo_device), dones.to(self.ppo_device), infos
        else:
            return self.obs_to_tensors(obs), torch.from_numpy(rewards).to(self.ppo_device).float(), torch.from_numpy(dones).to(self.ppo_device), infos

    def get_action_values(self, agent_id, obs):
        processed_obs = self._preproc_obs(obs['obs'])
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

    def set_eval(self):
        [self.models[agent_id].eval() for agent_id in range(self.num_multi_agents)]
        if self.normalize_rms_advantage:
            self.advantage_mean_std.eval()

    def set_train(self):
        [self.models[agent_id].train() for agent_id in range(self.num_multi_agents)]
        if self.normalize_rms_advantage:
            self.advantage_mean_std.train()

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
                    processed_obs = self._preproc_obs(obs['obs'])
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
                self.exp_buffs[agent_id].update_data('obses', n, self.obs['obs'+str(agent_id) if agent_id > 0 else 'obs'])
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
                    self.exp_buffs[agent_id].update_data('rewards', n, discounted_reward)

            # TODO, following codes need to be expanded to multi-agent settings
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices[::self.num_agents]

            for agent_id in range(self.num_multi_agents):
                self.current_rewards_list[agent_id] += rewards[:, agent_id].unsqueeze(-1)
                self.current_shaped_rewards_list[agent_id] += shaped_rewards[:, agent_id].unsqueeze(-1)

                self.game_rewards_list[agent_id].update(self.current_rewards_list[agent_id][env_done_indices])
                self.game_shaped_rewards_list[agent_id].update(self.current_shaped_rewards_list[agent_id][env_done_indices])

            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.dones.float()

            for agent_id in range(self.num_multi_agents):
                self.current_rewards_list[agent_id] = self.current_rewards_list[agent_id] * not_dones.unsqueeze(1)
                self.current_shaped_rewards_list[agent_id] = self.current_shaped_rewards_list[agent_id] * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        # TODO, get_values should be expanded to multi-agent
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

        self.set_train()

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)

        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()

        a_losses = []
        c_losses = []
        entropies = []
        kls = []
        if self.has_central_value:
            self.train_central_value()

        for mini_ep in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                a_loss, c_loss, entropy, kl, last_lr, lr_mul = self.train_actor_critic(self.dataset[i])
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)

            av_kls = torch_ext.mean_list(ep_kls)
            if self.multi_gpu:
                dist.all_reduce(av_kls, op=dist.ReduceOp.SUM)
                av_kls /= self.world_size

            self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0,
                                                                    av_kls.item())
            self.update_lr(self.last_lr)
            kls.append(av_kls)
            self.diagnostics.mini_epoch(self, mini_ep)
            if self.normalize_input:
                self.model.running_mean_std.eval()  # don't need to update statstics more than one miniepoch

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        return batch_dict[
            'step_time'], play_time, update_time, total_time, a_losses, c_losses, entropies, kls, last_lr, lr_mul

    def train(self):
        self.init_tensors()
        self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        self.obs, share_obs = self.env_reset()
        self.curr_frames = self.batch_size_envs

        if self.multi_gpu:
            print("====================broadcasting parameters")
            model_params = [self.model.state_dict()]
            dist.broadcast_object_list(model_params, 0)
            self.model.load_state_dict(model_params[0])

        while True:
            epoch_num = self.update_epoch()
            step_time, play_time, update_time, sum_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul = self.train_epoch()
            total_time += sum_time
            frame = self.frame // self.num_agents

            # cleaning memory to optimize space
            self.dataset.update_values_dict(None)
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
                                 a_losses, c_losses, entropies, kls, last_lr, lr_mul, frame,
                                 scaled_time, scaled_play_time, curr_frames)

                if len(b_losses) > 0:
                    self.writer.add_scalar('losses/bounds_loss', torch_ext.mean_list(b_losses).item(), frame)

                if self.has_soft_aug:
                    self.writer.add_scalar('losses/aug_loss', np.mean(aug_losses), frame)

                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_shaped_rewards = self.game_shaped_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()
                    self.mean_rewards = mean_rewards[0]

                    for i in range(self.value_size):
                        rewards_name = 'rewards' if i == 0 else 'rewards{0}'.format(i)
                        self.writer.add_scalar(rewards_name + '/step'.format(i), mean_rewards[i], frame)
                        self.writer.add_scalar(rewards_name + '/iter'.format(i), mean_rewards[i], epoch_num)
                        self.writer.add_scalar(rewards_name + '/time'.format(i), mean_rewards[i], total_time)
                        self.writer.add_scalar('shaped_' + rewards_name + '/step'.format(i), mean_shaped_rewards[i],
                                               frame)
                        self.writer.add_scalar('shaped_' + rewards_name + '/iter'.format(i), mean_shaped_rewards[i],
                                               epoch_num)
                        self.writer.add_scalar('shaped_' + rewards_name + '/time'.format(i), mean_shaped_rewards[i],
                                               total_time)

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
