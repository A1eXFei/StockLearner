import configparser
import os
import random

import numpy as np
import tensorflow as tf

from rl.tensorflow.v1.agent import RLBaseAgent
from rl.tensorflow.v1.dqn import ReplayBuffer, PrioritizedReplayBuffer
from rl.tensorflow.v1.ops import clipped_error
from util import model_util


# TODO: 1. configure parameters - DONE
#       2. Double Q - DONE
#       3. Duel DQN
#       4. Prioritized Replay Buffer - DONE
#       3. Summary
#       4. Checkpoint - DONE
#       5. Copy network vars from target to prediction - DONE
#       6. learning rate ops - DONE


class DQNAgent(RLBaseAgent):
    def __init__(self, config_file, sess):
        super(DQNAgent, self).__init__()

        self.sess = sess
        self._init_params(config_file)

        if self.prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(self.buffer_size, self.prioritized_replay_alpha)
        else:
            self.replay_buffer = ReplayBuffer(self.buffer_size)

        with tf.variable_scope("step"):
            self.step_op = tf.Variable(0, trainable=False, name="step")
            self.step_input = tf.placeholder(dtype=tf.int32, shape=None, name="step_input")
            self.step_assign_op = self.step_op.assign(self.step_input)

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.saver = tf.train.Saver(max_to_keep=self.max_to_save)

        self._build_dqn()

        if self.enable_summary:
            self.writer = tf.summary.FileWriter(self.summary_log_dir, sess.graph)
        else:
            self.writer = None

    def _init_params(self, config_file):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.config.read(self.config_file)

        self.buffer_size = self.config.getint("replay_buffer", "buffer_size")
        self.prioritized_replay = self.config.getboolean("replay_buffer", "prioritized_replay")
        if self.prioritized_replay:
            self.prioritized_replay_alpha = self.config.getfloat("replay_buffer", "alpha")
            self.prioritized_replay_beta = self.config.getfloat("replay_buffer", "beta")
            self.prioritized_replay_beta_incremental = self.config.getfloat("replay_buffer", "incremental")
            self.prioritized_replay_eps = self.config.getfloat("replay_buffer", "eps")

        self.network_config_file = self.config.get("network", "config_file")

        self.is_training = self.config.getboolean("train_ops", "is_training")
        self.learning_rate = self.config.getfloat("train_ops", "learning_rate")
        self.learning_rate_step = self.config.getint("train_ops", "learning_rate_step")
        self.learning_rate_decay_step = self.config.getint("train_ops", "learning_rate_decay_step")
        self.learning_rate_decay = self.config.getfloat("train_ops", "learning_rate_decay")
        self.learning_rate_minimum = self.config.getfloat("train_ops", "min_learning_rate")

        self.checkpoint_dir = self.config.get("train_ops", "model_dir")
        self.max_to_save = self.config.getint("train_ops", "max_to_save")
        self.save_frequency = self.config.getint("train_ops", "save_frequency")
        self.summary_log_dir = self.config.get("train_ops", "log_dir")
        self.enable_summary = self.config.getboolean("train_ops", "enable_log")

        self.learning_freq = self.config.getint("dqn", "learning_frequency")
        self.batch_size = self.config.getint("dqn", "batch_size")
        self.target_q_update_step = self.config.getint("dqn", "target_q_update_frequency")
        self.discount = self.config.getfloat("dqn", "discount")
        self.train_frequency = self.config.getint("dqn", "train_frequency")
        self.learn_start = self.config.getint("dqn", "learn_start")

        self.double_q = self.config.getboolean("dqn", "double_q")
        self.dueling = self.config.getboolean("dqn", "dueling")
        self.epsilon = self.config.getfloat("dqn", "epsilon")
        self.action_size = self.config.getint("dqn", "action_size")

    def _build_dqn(self):
        with tf.variable_scope("prediction"):
            print("#####################Building prediction network#####################")
            self.pred_network = model_util.get_network(self.network_config_file, network_name="prediction")
            self.s_t = self.pred_network.x
            self.q = self.pred_network.y

        self.q_action = tf.argmax(self.q, axis=1)

        with tf.variable_scope("target"):
            print("#####################Building target network#####################")
            self.target_network = model_util.get_network(self.network_config_file, network_name="target")
            self.target_s_t = self.target_network.x
            self.target_q = self.target_network.y

        self.target_q_idx = tf.placeholder(dtype=tf.int32, shape=[None, None], name='outputs_idx')
        self.target_q_with_idx = tf.gather_nd(self.target_q, self.target_q_idx)

        with tf.variable_scope('optimizer'):
            self.target_q_t = tf.placeholder(dtype=tf.float32, shape=[None], name='target_q_t')
            self.action = tf.placeholder(dtype=tf.int32, shape=[None], name='action')

            action_one_hot = tf.one_hot(self.action, self.action_size, 1.0, 0.0, name='action_one_hot')
            q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')

            self.delta = self.target_q_t - q_acted
            self.global_step = tf.Variable(0, trainable=False)
            self.loss = tf.reduce_mean(clipped_error(self.delta), name='loss')

            self.learning_rate_step = tf.placeholder(dtype=tf.int32, shape=None, name='learning_rate_step')
            self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
                                               tf.train.exponential_decay(
                                                   self.learning_rate,
                                                   self.learning_rate_step,
                                                   self.learning_rate_decay_step,
                                                   self.learning_rate_decay,
                                                   staircase=True))
            self.optimizer = tf.train.RMSPropOptimizer(
                self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss)

        tf.global_variables_initializer().run()
        self.load_model()
        self._update_target_q_network()

    def _update_target_q_network(self):
        variables_names = [v.name for v in tf.trainable_variables()]
        graph = tf.get_default_graph()
        for var in variables_names:
            if "prediction" in var:
                from_tensor = graph.get_tensor_by_name(var)
                to_tensor = graph.get_tensor_by_name(var.replace("prediction", "target"))
                copy_tensor = tf.assign(to_tensor, from_tensor)
                self.sess.run(copy_tensor)

    def save_model(self, step):
        print(" [*] Saving checkpoints...")
        self.saver.save(self.sess, os.path.join(self.checkpoint_dir, "dqn_agent"), global_step=step)

    def load_model(self):
        print(" [*] Loading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            fname = os.path.join(self.checkpoint_dir, ckpt_name)
            self.saver.restore(self.sess, fname)
            print(" [*] Load SUCCESS: %s" % fname)
            return True
        else:
            print(" [!] Load FAILED: %s" % self.checkpoint_dir)
            return False

    def study(self, global_step):
        if global_step > self.learn_start:
            if global_step % self.train_frequency == 0:
                self.q_learning_mini_batch(global_step)

            if global_step % self.target_q_update_step == 0:
                # print(global_step)
                self._update_target_q_network()

            if global_step % self.save_frequency == 0:
                self.save_model(global_step)

    def q_learning_mini_batch(self, step):
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

    def store(self, observe, reward, action, next_observe, terminal):
        if observe is None or next_observe is None:
            return
        self.replay_buffer.add(observe, reward, action, next_observe, terminal)

    def choose_action(self, s_t, test_ep=None):
        if random.random() < self.epsilon:
            action = random.randrange(self.action_size)
        else:
            action = self.q_action.eval({self.s_t: [s_t], self.pred_network.is_training: self.is_training})[0]

        self.epsilon = max(self.epsilon - 0.00001, 0.1)
        return action
