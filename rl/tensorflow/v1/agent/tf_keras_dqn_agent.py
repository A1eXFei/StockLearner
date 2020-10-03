import os
import random

import numpy as np
import tensorflow as tf

from keras.model.tree_model_builder import TreeModelBuilder
from rl.tensorflow.v1.agent import RLBaseAgent
from rl.tensorflow.v1.dqn import ReplayBuffer, PrioritizedReplayBuffer


class DQNAgent(RLBaseAgent):
    def __init__(self, yaml_config):
        super(DQNAgent, self).__init__()

        self.yaml_config = yaml_config

        self._init_params()

        self.network_yaml_config_file = "../../config_file/yaml_config/stock_mlp_baseline.yaml"

        if self.prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(self.buffer_size, self.prioritized_replay_alpha)
        else:
            self.replay_buffer = ReplayBuffer(self.buffer_size)

        with tf.variable_scope("step"):
            self.step_op = tf.Variable(0, trainable=False, name="step")
            self.step_input = tf.placeholder(dtype=tf.int32, shape=None, name="step_input")
            self.step_assign_op = self.step_op.assign(self.step_input)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # self.saver = tf.train.Saver(max_to_keep=self.max_to_save)

        self._build_dqn()

        if self.enable_tensorboard:
            self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, batch_size=self.batch_size)
        else:
            self.tensorboard_callback = None

    def _init_params(self):
        self.learning_freq = self.yaml_config["dqn"]["learning_frequency"]
        self.batch_size = self.yaml_config["dqn"]["batch_size"]
        self.target_q_update_step = self.yaml_config["dqn"]["target_q_update_frequency"]
        self.discount = self.yaml_config["dqn"]["discount"]
        self.train_frequency = self.yaml_config["dqn"]["train_frequency"]
        self.learn_start = self.yaml_config["dqn"]["learn_start"]

        self.double_q = self.yaml_config["dqn"]["double_q"]
        self.dueling = self.yaml_config["dqn"]["dueling"]
        self.epsilon = self.yaml_config["dqn"]["epsilon"]
        self.action_size = self.yaml_config["dqn"]["action_size"]

        self.buffer_size = self.yaml_config["dqn"]["replay_buffer"]["buffer_size"]
        self.prioritized_replay = self.yaml_config["dqn"]["replay_buffer"]["prioritized_replay"]

        if self.prioritized_replay:
            self.prioritized_replay_alpha = self.yaml_config["dqn"]["replay_buffer"]["alpha"]
            self.prioritized_replay_beta = self.yaml_config["dqn"]["replay_buffer"]["beta"]
            self.prioritized_replay_beta_incremental = self.yaml_config["dqn"]["replay_buffer"]["incremental"]
            self.prioritized_replay_eps = self.yaml_config["dqn"]["replay_buffer"]["eps"]

        self.network_yaml_config_file = self.yaml_config["model"]["config_path"]
        self.output_dir = self.yaml_config["model"]["output_dir"]
        self.max_to_save = self.yaml_config["model"]["max_to_save"]
        self.save_frequency = self.yaml_config["model"]["save_frequency"]
        self.log_dir = self.yaml_config["model"]["log_dir"]
        self.enable_tensorboard = self.yaml_config["model"]["enable_tensorboard"]

        if self.yaml_config["run"]["for"] == "train":
            self.is_training = True
            self.learning_rate = self.yaml_config["run"]["train"]["learning_rate"]
            self.learning_rate_step = self.yaml_config["run"]["train"]["learning_rate_step"]
            self.learning_rate_decay_step = self.yaml_config["run"]["train"]["learning_rate_decay_step"]
            self.learning_rate_decay = self.yaml_config["run"]["train"]["learning_rate_decay"]
            self.learning_rate_minimum = self.yaml_config["run"]["train"]["min_learning_rate"]
        else:
            self.is_training = False

        self.model_file = os.path.join(self.output_dir, "DQN_Agent.h5")

    def _build_dqn(self):
        self.q_network = TreeModelBuilder(self.network_yaml_config_file, "q_network").get_model()
        self.target_q_network = TreeModelBuilder(self.network_yaml_config_file, "target_q_network").get_model()

        if os.path.exists(self.model_file):
            self.__load_model()

    def choose_action(self, observation):
        # TODO: to review
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.q_network.predict(observation)
        print(act_values)
        return np.argmax(act_values[0]), act_values  # returns action

    def store(self, observe, reward, action, next_observe, terminal):
        if observe is None or next_observe is None:
            return
        self.replay_buffer.add(observe, reward, action, next_observe, terminal)

    def study(self, global_step):
        if global_step > self.learn_start:
            if global_step % self.train_frequency == 0:
                self._q_learning_mini_batch(global_step)

            if global_step % self.target_q_update_step == 0:
                self._update_target_q_network()

            if global_step % self.save_frequency == 0:
                self._save_model()

    def _q_learning_mini_batch(self, step):
        # TODO: wip
        if len(self.replay_buffer) < self.batch_size:
            return

        if self.prioritized_replay:
            self.prioritized_replay_beta = np.min([1.,
                                                   self.prioritized_replay_beta
                                                   + self.prioritized_replay_beta_incremental])
            s_t, action, reward, s_t_plus_1, terminal, weights, batch_idxes \
                = self.replay_buffer.sample(self.batch_size, self.prioritized_replay_beta)
        else:
            s_t, action, reward, s_t_plus_1, terminal = self.replay_buffer.sample(self.batch_size)

        if self.double_q:
            # Double Q-learning
            pred_action = self.q_action.eval({self.s_t: s_t_plus_1,
                                              self.pred_network.is_training: self.is_training})
            q_t_plus_1_with_pred_action = self.target_q_with_idx.eval({self.target_s_t: s_t_plus_1,
                                                                       self.target_q_idx: [[idx, pred_a] for idx, pred_a in enumerate(pred_action)],
                                                                       self.target_network.is_training: self.is_training})
            target_q_t = (1. - terminal) * self.discount * q_t_plus_1_with_pred_action + reward
        else:
            q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1,
                                             self.target_network.is_training: self.is_training})
            terminal = np.array(terminal) + 0.
            max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
            target_q_t = (1. - terminal) * self.discount * max_q_t_plus_1 + reward

        _, q_t, loss, td_errors = self.sess.run([self.optimizer, self.q, self.loss, self.delta],
                                                feed_dict={self.target_q_t: target_q_t,
                                                           self.action: action,
                                                           self.s_t: s_t,
                                                           self.learning_rate_step: step,
                                                           self.pred_network.is_training: self.is_training
                                                           })

        if self.prioritized_replay:
            new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
            self.replay_buffer.update_priorities(batch_idxes, new_priorities)

    def _load_model(self):
        self.target_q_network = tf.keras.models.load_model(filepath=self.model_file)
        self.__update_target_q_network()

    def _save_model(self):
        tf.keras.models.save_model(self.q_network, filepath=self.model_file)

    def _update_target_q_network(self):
        self.q_network.set_weights(self.target_q_network.get_weights())


if __name__ == "__main__":
    import yaml

    APP_CONFIG_FILE_PATH = "../../tf_keras_rl_ops.yaml"
    f = open(APP_CONFIG_FILE_PATH, 'r', encoding='utf-8')
    c = yaml.load(f)
    agent = DQNAgent(c)
