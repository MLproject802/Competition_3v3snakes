import os
import torch
import numpy as np
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from common import soft_update, hard_update, device

from algo.ppo_net import PPOActor
from algo.ppo_net import PPOCritic

from torch.distributions import Categorical


class MAPPO:
    def __init__(self, obs_dim, act_dim, num_agent, args):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agent = num_agent
        self.device = device
        self.a_lr = args.a_lr
        self.c_lr = args.c_lr
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.args = args
        # TODO: self.state_shape = args.state_shape
        self.n_actions = args.n_actions
        actor_input_shape = self.obs_dim
        critic_input_shape = self._get_critic_input_shape()

        # if args.last_action:
        #     actor_input_shape += self.n_actions
        actor_input_shape += self.num_agent

        self.actor = PPOActor(actor_input_shape, args).to(self.device)
        self.critic = PPOCritic(critic_input_shape, self.args).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.c_lr)

        self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map

        self.actor_hidden = None
        self.critic_hidden = None

    def _get_critic_input_shape(self):
        # state
        input_shape = self.state_shape
        # obs
        input_shape += self.obs_dim
        # agent_id
        input_shape += self.num_agent

        # input_shape += self.n_actions * self.num_agent * 2  # 54
        return input_shape

    def learn(self, batch, max_episode_len, train_step, time_steps=0):
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        u, r, avail_u, terminated, s = batch['u'], batch['r'],  batch['avail_u'], batch['terminated'], batch['s']

        mask = (1 - batch["padded"].float())

        u = u.cuda().to(self.device)
        mask = mask.cuda().to(self.device)
        r = r.cuda().to(self.device)
        terminated = terminated.cuda().to(self.device)
        s = s.cuda().to(self.device)

        mask = mask.repeat(1, 1, self.num_agent)
        r = r.repeat(1, 1, self.num_agent)
        terminated = terminated.repeat(1, 1, self.num_agent)

        old_values, _ = self._get_values(batch, max_episode_len)
        old_values = old_values.squeeze(dim=-1)
        old_action_prob = self._get_action_prob(batch, max_episode_len)

        old_dist = Categorical(old_action_prob)
        old_log_pi_taken = old_dist.log_prob(u.squeeze(dim=-1))
        old_log_pi_taken[mask == 0] = 0.0

        for _ in range(self.args.ppo_n_epochs):
            self.init_hidden(episode_num)

            values, target_values = self._get_values(batch, max_episode_len)
            values = values.squeeze(dim=-1)

            returns = torch.zeros_like(r)
            deltas = torch.zeros_like(r)
            advantages = torch.zeros_like(r)

            prev_return = 0.0
            prev_value = 0.0
            prev_advantage = 0.0
            for transition_idx in reversed(range(max_episode_len)):
                returns[:,transition_idx] = r[:,transition_idx] + self.args.gamma * prev_return * (1-terminated[:,transition_idx]) * mask[:, transition_idx]
                deltas[:,transition_idx] = r[:,transition_idx] + self.args.gamma * prev_value * (1-terminated[:,transition_idx]) * mask[:, transition_idx]\
                                           - values[:, transition_idx]
                advantages[:,transition_idx] = deltas[:,transition_idx] + self.args.gamma * self.args.lamda * prev_advantage * (1-terminated[:,transition_idx]) * mask[:, transition_idx]

                prev_return = returns[:,transition_idx]
                prev_value = values[:,transition_idx]
                prev_advantage = advantages[:,transition_idx]

            advantages = (advantages - advantages.mean()) / ( advantages.std() + 1e-8)
            advantages = advantages.to(self.device)

            action_prob = self._get_action_prob(batch, max_episode_len)
            dist = Categorical(action_prob)
            log_pi_taken = dist.log_prob(u.squeeze(dim=-1))
            log_pi_taken[mask == 0] = 0.0

            ratios = torch.exp(log_pi_taken - old_log_pi_taken.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.args.clip_param, 1.0 + self.args.clip_param) * advantages
            entropy = dist.entropy()
            entropy[mask == 0] = 0.0
            actor_loss = torch.min(surr1, surr2) + self.args.entropy_coeff * entropy
            actor_loss = - (actor_loss * mask).sum() / mask.sum()
            
            error_clip = torch.clamp(values - old_values.detach(), -self.args.clip_param, self.args.clip_param) + old_values.detach() - returns
            error_original = values - returns
            critic_loss = 0.5 * torch.max(error_original**2, error_clip**2)
            critic_loss = (mask * critic_loss).sum() / mask.sum()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.grad_norm_clip)
            self.actor_optimizer.step()
    
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()



    def _get_critic_inputs(self, batch, transition_idx, max_episode_len):
        obs, obs_next, s, s_next = batch['o'][:, transition_idx], batch['o_next'][:, transition_idx],\
                                   batch['s'][:, transition_idx], batch['s_next'][:, transition_idx]
        s = s.unsqueeze(1).expand(-1, self.num_agent, -1)
        s_next = s_next.unsqueeze(1).expand(-1, self.num_agent, -1)
        episode_num = obs.shape[0]

        inputs, inputs_next = [], []

        inputs.append(s)
        inputs_next.append(s_next)

        inputs.append(obs)
        inputs_next.append(obs_next)

        inputs.append(torch.eye(self.num_agent).unsqueeze(0).expand(episode_num, -1, -1))
        inputs_next.append(torch.eye(self.num_agent).unsqueeze(0).expand(episode_num, -1, -1))

        inputs = torch.cat([x.reshape(episode_num * self.num_agent, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.num_agent, -1) for x in inputs_next], dim=1)

        return inputs, inputs_next

    def _get_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        v_evals, v_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_critic_inputs(batch, transition_idx, max_episode_len)
            inputs = inputs.to(self.device)
            self.critic_hidden = self.critic_hidden.to(self.device)

            v_eval, self.critic_hidden = self.critic(inputs, self.critic_hidden)
            v_eval = v_eval.view(episode_num, self.num_agent, -1)
            v_evals.append(v_eval)

        v_evals = torch.stack(v_evals, dim=1)
        return v_evals, v_targets

    def _get_actor_inputs(self, batch, transition_idx):
        obs, u_onehot = batch['o'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs = []
        inputs.append(obs)

        if self.args.last_action:
            if transition_idx == 0:
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
        if self.args.reuse_network:
            inputs.append(torch.eye(self.args.num_agent).unsqueeze(0).expand(episode_num, -1, -1))

        inputs = torch.cat([x.reshape(episode_num * self.args.num_agent, -1) for x in inputs], dim=1)

        return inputs

    def _get_action_prob(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        avail_actions = batch['avail_u']
        action_prob = []
        for transition_idx in range(max_episode_len):
            inputs = self._get_actor_inputs(batch, transition_idx)
            if self.args.use_gpu:
                inputs = inputs.cuda()
                self.actor_hidden = self.actor_hidden.cuda()
            outputs, self.actor_hidden = self.actor(inputs, self.actor_hidden)
            outputs = outputs.view(episode_num, self.num_agent, -1)
            prob = torch.nn.functional.softmax(outputs, dim=-1)
            action_prob.append(prob)

        action_prob = torch.stack(action_prob, dim=1).cpu()
        action_prob = action_prob + 1e-10

        action_prob[avail_actions == 0] = 0.0
        action_prob = action_prob / action_prob.sum(dim=-1, keepdim=True)
        action_prob[avail_actions == 0] = 0.0
        
        action_prob = action_prob + 1e-10
        action_prob = action_prob.to(self.device)
        return action_prob

    def init_hidden(self, episode_num):
        self.actor_hidden = torch.zeros((episode_num, self.num_agent, self.args.rnn_hidden_dim))
        self.critic_hidden = torch.zeros((episode_num, self.num_agent, self.args.rnn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.critic.state_dict(), self.model_dir + '/' + num + '_critic_params.pkl')
        torch.save(self.actor.state_dict(),  self.model_dir + '/' + num + '_actor_params.pkl')