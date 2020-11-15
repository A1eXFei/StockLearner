import torch
import math
import random
import torch.nn.functional as F
from torch import optim
from rl.pytorch.agents.base_agent import BaseAgent
from rl.pytorch.agents.replay_buffer import Transition


class DQN(BaseAgent):
    def __init__(self, n_actions, q_network, replay_buffer, batch_size=128,
                 eps_start=0.9, eps_end=0.05, eps_decay=200, gamma=1.0, learning_rate=0.0001):
        BaseAgent.__init__(self, n_actions, batch_size)
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.steps_done = 0
        self.memory = replay_buffer
        self.policy_net = q_network
        self.target_net = q_network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), self.learning_rate)

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.compute_state_action_values(state_batch, action_batch)
        expected_state_action_values = self.compute_expected_state_action_values(batch, reward_batch)
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.take_optimisation_step(self.optimizer, loss, self.policy_net)

    def compute_state_action_values(self, state_batch, action_batch):
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        return state_action_values

    def compute_expected_state_action_values(self, batch, reward_batch):
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        next_state_values = self.compute_next_state_action_values(non_final_mask, non_final_next_states)
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        return expected_state_action_values

    def compute_next_state_action_values(self, non_final_mask, non_final_next_states):
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        return next_state_values

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


class DDQN(DQN):
    def __init__(self, n_actions, q_network, replay_buffer):
        DQN.__init__(self, n_actions, q_network, replay_buffer)

    def compute_next_state_action_values(self, non_final_mask, non_final_next_states):
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        max_action_indexes = self.policy_net(non_final_next_states).detach().argmax(1)
        # print("=====")
        # print(next_state_values.shape)
        # print(max_action_indexes.shape)
        # print(self.target_net(non_final_next_states).shape)
        # print(self.target_net(non_final_next_states).max(1)[0].detach().shape)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach().gather(0, max_action_indexes)
        return next_state_values
