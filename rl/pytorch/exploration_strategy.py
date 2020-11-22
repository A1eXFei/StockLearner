import numpy as np
import random
import torch


class BaseExplorationStrategy(object):
    """Base abstract class for agent exploration strategies. Every exploration strategy must inherit from this class
    and implement the methods perturb_action_for_exploration_purposes and add_exploration_rewards"""
    def __init__(self):
        pass

    def perturb_action_for_exploration_purposes(self, action_info):
        """Perturbs the action of the agent to encourage exploration"""
        raise ValueError("Must be implemented")

    def add_exploration_rewards(self, reward_info):
        """Actions intrinsic rewards to encourage exploration"""
        raise ValueError("Must be implemented")

    def reset(self):
        """Resets the noise process"""
        raise ValueError("Must be implemented")


class EpsilonGreedyExploration(BaseExplorationStrategy):
    """Implements an epsilon greedy exploration strategy"""
    def __init__(self, exploration_cycle_episodes_length, random_episodes_to_run, epsilon_decay_rate_denominator):
        super().__init__()
        self.epsilon_decay_rate_denominator = epsilon_decay_rate_denominator
        self.notified_that_exploration_turned_off = False
        self.exploration_cycle_episodes_length = exploration_cycle_episodes_length
        self.random_episodes_to_run = random_episodes_to_run

    def perturb_action_for_exploration_purposes(self, action_info):
        """Perturbs the action of the agent to encourage exploration"""
        action_values = action_info["action_values"]
        turn_off_exploration = action_info["turn_off_exploration"]
        episode_number = action_info["episode_number"]
        if turn_off_exploration and not self.notified_that_exploration_turned_off:
            print(" ")
            print("Exploration has been turned OFF")
            print(" ")
            self.notified_that_exploration_turned_off = True
        epsilon = self.get_updated_epsilon_exploration(action_info)

        if (random.random() > epsilon or turn_off_exploration) and (episode_number >= self.random_episodes_to_run):
            return torch.argmax(action_values).item()
        return  np.random.randint(0, action_values.shape[1])

    def get_updated_epsilon_exploration(self, action_info, epsilon=1.0):
        """
        Gets the probability that we just pick a random action.
        This probability decays the more episodes we have seen
        """
        episode_number = action_info["episode_number"]
        epsilon_decay_denominator = self.epsilon_decay_rate_denominator

        if self.exploration_cycle_episodes_length is None:
            epsilon = epsilon / (1.0 + (episode_number / epsilon_decay_denominator))
        else:
            epsilon = self.calculate_epsilon_with_cyclical_strategy(episode_number)
        return epsilon

    def calculate_epsilon_with_cyclical_strategy(self, episode_number):
        """Calculates epsilon according to a cyclical strategy"""
        max_epsilon = 0.5
        min_epsilon = 0.001
        increment = (max_epsilon - min_epsilon) / float(self.exploration_cycle_episodes_length / 2)
        cycle = [ix for ix in range(int(self.exploration_cycle_episodes_length / 2))] + [ix for ix in range(
            int(self.exploration_cycle_episodes_length / 2), 0, -1)]
        cycle_ix = episode_number % self.exploration_cycle_episodes_length
        epsilon = max_epsilon - cycle[cycle_ix] * increment
        return epsilon

    def add_exploration_rewards(self, reward_info):
        """Actions intrinsic rewards to encourage exploration"""
        return reward_info["reward"]

    def reset(self):
        """Resets the noise process"""
        pass
