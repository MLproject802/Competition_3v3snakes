import os
import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from replay_buffer import ReplayBuffer
from common import soft_update, hard_update, device
from algo.network import SACActor, SACCritic
from torch.distributions import Categorical
import copy
from collections import OrderedDict

def get_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.to("cpu").detach().numpy()

class SAC:

    def __init__(self, obs_dim, act_dim, num_agent, args):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agent = num_agent
        self.device = device
        self.a_lr = args.a_lr
        self.c_lr = args.c_lr
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.model_episode = args.model_episode
        self.eps = args.epsilon
        self.decay_speed = args.epsilon_speed
        self.output_activation = args.output_activation
        self.log_alpha = torch.tensor(
            np.log(args.alpha), requires_grad=True, device=device
        )
        self.reward_scale = args.reward_scale
        self.prev_env_step = 0
        self.total_env_step = 0
        self.length_adv = 0
        self.prev_opp_segments = None
        self.prev_opp_length = 9

        # Initialise actor network and critic network with ξ and θ
        self.actor = SACActor(obs_dim, act_dim, num_agent, args, self.output_activation).to(self.device)
        self.qf1 = SACCritic(obs_dim, act_dim, num_agent, args).to(self.device)
        self.qf2 = SACCritic(obs_dim, act_dim, num_agent, args).to(self.device)
        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)

        self.policy_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)
        self.qf1_optimizer = torch.optim.Adam(self.qf1.parameters(), lr=self.c_lr)
        self.qf2_optimizer = torch.optim.Adam(self.qf2.parameters(), lr=self.c_lr)

        # Initialise replay buffer R
        self.replay_buffer = ReplayBuffer(args.buffer_size, args.batch_size)

        self.eval_statistics = OrderedDict()

    # Random process N using epsilon greedy
    def choose_action(self, obs, evaluation=False):

        p = np.random.random()
        if p > self.eps or evaluation:
            obs = torch.Tensor([obs]).to(self.device)
            action = self.actor(obs).cpu().detach().numpy()[0]
        else:
            action = self.random_action()

        self.eps *= self.decay_speed
        return action

    def random_action(self):
        if self.output_activation == 'tanh':
            return np.random.uniform(low=-1, high=1, size=self.act_dim**self.num_agent)
        return np.random.uniform(low=0, high=1, size=self.act_dim**self.num_agent)

    def update(self):

        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample a greedy_min mini-batch of M transitions from R
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.get_batches()

        state_batch = torch.Tensor(state_batch).reshape(self.batch_size, self.num_agent, -1).to(self.device)
        action_batch = torch.Tensor(action_batch).reshape(self.batch_size, 1).to(self.device, dtype=torch.int64)
        reward_batch = self.reward_scale * torch.Tensor(reward_batch).reshape(self.batch_size, 1).to(self.device)
        next_state_batch = torch.Tensor(next_state_batch).reshape(self.batch_size, self.num_agent, -1).to(self.device)
        done_batch = torch.Tensor(done_batch).reshape(self.batch_size, 1).to(self.device)
        
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        q1_pred = self.qf1(next_state_batch).gather(1, action_batch)
        q2_pred = self.qf2(next_state_batch).gather(1, action_batch)
        next_action_dist = Categorical(self.actor(next_state_batch))
        next_q = next_action_dist.probs * torch.min(
            self.target_qf1(next_state_batch),
            self.target_qf2(next_state_batch),
        )
        target_v_values = next_q.sum(dim=-1) + self.alpha * next_action_dist.entropy()
        q_target = reward_batch + (
            1.0 - done_batch
        ) * self.gamma * target_v_values.unsqueeze(
            -1
        )  # original implementation has detach
        q_target = q_target.detach()
        qf1_loss = 0.5 * torch.mean((q1_pred - q_target) ** 2)
        qf2_loss = 0.5 * torch.mean((q2_pred - q_target) ** 2)

        qf1_loss.backward()
        qf2_loss.backward()

        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        # print(self.alpha * next_action_dist.entropy())
        # print(reward_batch)
        # torch.set_printoptions(threshold=10000)
        # print(next_action_dist.probs)
        """
        Policy Loss
        """
        action_dist = Categorical(probs=self.actor(state_batch))
        current_q = torch.min(
            self.qf1(state_batch),
            self.qf2(state_batch),
        )
        current_q = current_q.detach()

        self.policy_optimizer.zero_grad()
        policy_loss = -torch.mean(
            self.alpha * action_dist.entropy()
            + (action_dist.probs * current_q).sum(dim=-1)
        )
        policy_loss.backward()
        self.policy_optimizer.step()

        
        """
        Update networks
        """

        soft_update(self.qf1, self.target_qf1, self.tau)
        soft_update(self.qf2, self.target_qf2, self.tau)

        self.eval_statistics["alpha"] = self.alpha
        self.eval_statistics["entropy"] = np.mean(get_numpy(action_dist.entropy()))
        self.eval_statistics["QF1 Loss"] = np.mean(get_numpy(qf1_loss))
        self.eval_statistics["QF2 Loss"] = np.mean(get_numpy(qf2_loss))
        self.eval_statistics["Policy Loss"] = np.mean(get_numpy(policy_loss))
        self.eval_statistics["Q1 Predictions"] = np.mean(get_numpy(q1_pred))
        self.eval_statistics["Q2 Predictions"] = np.mean(get_numpy(q2_pred))

        return self.eval_statistics

    def get_loss(self):
        return self.c_loss, self.a_loss

    def load_model(self, run_dir, episode):
        print(f'\nBegin to load model: ')
        base_path = os.path.join(run_dir, 'trained_model')
        model_actor_path = os.path.join(base_path, "actor_" + str(episode) + ".pth")
        model_critic_path = os.path.join(base_path, "critic_" + str(episode) + ".pth")
        print(f'Actor path: {model_actor_path}')
        print(f'Critic path: {model_critic_path}')

        if os.path.exists(model_critic_path) and os.path.exists(model_actor_path):
            actor = torch.load(model_actor_path, map_location=device)
            critic = torch.load(model_critic_path, map_location=device)
            self.actor.load_state_dict(actor)
            self.critic.load_state_dict(critic)
            print("Model loaded!")
        else:
            sys.exit(f'Model not founded!')

    def save_model(self, run_dir, episode):
        base_path = os.path.join(run_dir, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_actor_path = os.path.join(base_path, "actor_" + str(episode) + ".pth")
        torch.save(self.actor.state_dict(), model_actor_path)

        model_critic_path = os.path.join(base_path, "critic_" + str(episode) + ".pth")
        torch.save(self.critic.state_dict(), model_critic_path)
