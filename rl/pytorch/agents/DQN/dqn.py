import torch
import torch.nn.functional as F
import numpy as np
from torch import optim
from rl.pytorch.agents.base_agent import BaseAgent
from rl.pytorch.exploration_strategy import EpsilonGreedyExploration


class DQN(BaseAgent):
    """A deep Q learning agent"""
    agent_name = "DQN"

    def __init__(self, network, acton_size, state_size, batch_size, replay_buffer, learning_rate, discount_rate,
                 exploration_cycle_episodes_length, random_episodes_to_run, epsilon_decay_rate_denominator,
                 gradient_clipping_norm, seed, score_required_to_win, visualise_individual_results, debug_mode, device):
        BaseAgent.__init__(self, seed, acton_size, state_size, score_required_to_win, batch_size,
                           visualise_individual_results, debug_mode, device)
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.gradient_clipping_norm = gradient_clipping_norm
        self.memory = replay_buffer # ReplayBuffer(buffer_size, batch_size, seed)
        self.q_network_local = network
        self.q_network_optimizer = optim.Adam(self.q_network_local.parameters(),
                                              lr=self.learning_rate, eps=1e-4)
        self.exploration_strategy = EpsilonGreedyExploration(exploration_cycle_episodes_length,
                                                             random_episodes_to_run,
                                                             epsilon_decay_rate_denominator)

    def reset_game(self):
        super(DQN, self).reset_game()
        self.update_learning_rate(self.learning_rate, self.q_network_optimizer)

    def select_action(self, state):
        """Uses the local Q network and an epsilon greedy policy to pick an action"""
        # PyTorch only accepts mini-batches and not single observations so we have to use unsqueeze to add
        # a "fake" dimension to make it a mini-batch rather than a single observation
        if state is None:
            return

        if isinstance(state, np.int64) or isinstance(state, int):
            state = np.array([state])

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        if len(state.shape) < 2:
            state = state.unsqueeze(0)
        # puts network in evaluation mode
        self.q_network_local.eval()
        with torch.no_grad():
            action_values = self.q_network_local(state)

        # puts network back in training mode
        self.q_network_local.train()
        action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action_values": action_values,
                                                                                    "turn_off_exploration": self.turn_off_exploration,                                                                           "episode_number": self.episode_number})
        self.logger.debug("Q values {} -- Action chosen {}".format(action_values, action))
        self.steps_done = self.steps_done + 1
        return action

    def learn(self, experiences=None):
        """Runs a learning iteration for the Q network"""
        if experiences is None:
            states, actions, rewards, next_states, dones = self.sample_experiences()
        else:
            states, actions, rewards, next_states, dones = experiences
        loss = self.compute_loss(states, next_states, rewards, actions, dones)

        # self.logger.info("Action counts {}".format(Counter(actions_list)))
        self.take_optimisation_step(self.q_network_optimizer, self.q_network_local, loss, self.gradient_clipping_norm)

    def compute_loss(self, states, next_states, rewards, actions, dones):
        """Computes the loss required to train the Q network"""
        with torch.no_grad():
            Q_targets = self.compute_q_targets(next_states, rewards, dones)
        Q_expected = self.compute_expected_q_values(states, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        return loss

    def compute_q_targets(self, next_states, rewards, dones):
        """Computes the q_targets we will compare to predicted q values to create the loss to train the Q network"""
        Q_targets_next = self.compute_q_values_for_next_states(next_states)
        Q_targets = self.compute_q_values_for_current_states(rewards, Q_targets_next, dones)
        return Q_targets

    def compute_q_values_for_next_states(self, next_states):
        """Computes the q_values for next state we will use to create the loss to train the Q network"""
        Q_targets_next = self.q_network_local(next_states).detach().max(1)[0].unsqueeze(1)
        return Q_targets_next

    def compute_q_values_for_current_states(self, rewards, Q_targets_next, dones):
        """Computes the q_values for current state we will use to create the loss to train the Q network"""
        Q_targets_current = rewards + (self.discount_rate * Q_targets_next * (1 - dones))
        return Q_targets_current

    def compute_expected_q_values(self, states, actions):
        """Computes the expected q_values we will use to create the loss to train the Q network"""
        # must convert actions to long so can be used as index
        Q_expected = self.q_network_local(states).gather(1, actions.long())
        return Q_expected

    def locally_save_policy(self):
        """Saves the policy"""
        torch.save(self.q_network_local.state_dict(), "Models/{}_local_network.pt".format(self.agent_name))

    def time_for_q_network_to_learn(self):
        """Returns boolean indicating whether enough steps have been taken for learning to begin and there are
        enough experiences in the replay buffer to learn from"""
        return self.right_amount_of_steps_taken() and self.enough_experiences_to_learn_from()

    def right_amount_of_steps_taken(self, global_step_number, update_every_n_steps):
        """Returns boolean indicating whether enough steps have been taken for learning to begin"""
        return global_step_number % update_every_n_steps == 0

    def sample_experiences(self):
        """Draws a random sample of experience from the memory buffer"""
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences
        return states, actions, rewards, next_states, dones


class DQNWithFixedQTargets(DQN):
    """A DQN agent that uses an older version of the q_network as the target network"""
    agent_name = "DQN with Fixed Q Targets"

    def __init__(self, network, acton_size, state_size, batch_size, replay_buffer, learning_rate, discount_rate, tau,
                 exploration_cycle_episodes_length, random_episodes_to_run, epsilon_decay_rate_denominator,
                 gradient_clipping_norm, seed, score_required_to_win, visualise_individual_results, debug_mode, device):
        DQN.__init__(self, network, acton_size, state_size, batch_size, replay_buffer, learning_rate, discount_rate,
                     exploration_cycle_episodes_length, random_episodes_to_run, epsilon_decay_rate_denominator,
                     gradient_clipping_norm, seed, score_required_to_win, visualise_individual_results,
                     debug_mode, device)
        self.tau = tau
        self.q_network_target = network
        BaseAgent.copy_model_over(from_model=self.q_network_local, to_model=self.q_network_target)

    def learn(self, experiences=None):
        """Runs a learning iteration for the Q network"""
        super(DQNWithFixedQTargets, self).learn(experiences=experiences)
        self.soft_update_of_target_network(self.q_network_local, self.q_network_target,
                                           self.tau)  # Update the target network

    def compute_q_values_for_next_states(self, next_states):
        """Computes the q_values for next state we will use to create the loss to train the Q network"""
        Q_targets_next = self.q_network_target(next_states).detach().max(1)[0].unsqueeze(1)
        return Q_targets_next