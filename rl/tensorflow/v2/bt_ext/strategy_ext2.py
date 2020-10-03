# -*- coding: UTF-8 -*-
from queue import Queue

import backtrader as bt
import numpy as np
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.networks import actor_distribution_network, q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import array_spec, tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import common


class RLCommonStrategy(bt.Strategy):
    params = (
        ("printlog", False),
    )

    def log(self, txt, dt=None, doprint=False):
        """ Logging function fot this strategy"""
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print("%s, %s" % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the schema[0] dataseries
        self.log("init strategy", doprint=False)
        self.min_time_step = 5
        self.time_step = 1
        self.dataclose = self.datas[0].close
        self.p_change = self.datas[0].p_change

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.date = None
        self.first_date = None
        self.last_date = None

        self.feature_columns = []
        feature_columns = self.datas[0].columns[1:-1]
        for feature in feature_columns:
            self.feature_columns.append(getattr(self.datas[0], feature.lower()))

        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
        self._action_spec = tensor_spec.from_spec(self._action_spec)
        self._observation_spec = array_spec.ArraySpec(shape=(len(feature_columns),), dtype=np.float32,
                                                      name='observation')
        self._observation_spec = tensor_spec.from_spec(self._observation_spec)

        self.agent = None
        self.actor_net = None
        self.replay_buffer = None
        self.learning_rate = 1e-3
        self.train_step_counter = tf.compat.v2.Variable(0)

        self._current_time_step = None
        self._last_time_step = None
        self._action_step = None
        self._last_action_step = None

        self.value_queue = Queue(maxsize=2)
        self.build_actor_agent()
        # self.build_dqn_agent()
        self.build_replay_buffer()

    def build_actor_agent(self):
        if self.env.get_agent() is None:
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.actor_net = actor_distribution_network.ActorDistributionNetwork(
                tensor_spec.from_spec(self.observation_spec()),
                tensor_spec.from_spec(self.action_spec()),
                fc_layer_params=(256, 512, 512, 128)
            )

            self.agent = reinforce_agent.ReinforceAgent(
                tensor_spec.from_spec(self.time_step_spec()),
                tensor_spec.from_spec(self.action_spec()),
                actor_network=self.actor_net,
                optimizer=optimizer,
                normalize_returns=True,
                train_step_counter=self.train_step_counter
            )

            self.agent.initialize()
            self.agent.train = common.function(self.agent.train)
        else:
            self.agent = self.env.get_agent()

    def build_dqn_agent(self):
        if self.env.get_agent() is None:
            fc_layer_params = (256, 512, 512, 128)
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)

            q_net = q_network.QNetwork(
                tensor_spec.from_spec(self.observation_spec()),
                tensor_spec.from_spec(self.action_spec())
            )

            self.agent = dqn_agent.DdqnAgent(
                tensor_spec.from_spec(self.time_step_spec()),
                tensor_spec.from_spec(self.action_spec()),
                q_network=q_net,
                optimizer=optimizer,
                td_errors_loss_fn=common.element_wise_squared_loss,
                train_step_counter=self.train_step_counter
            )

            self.agent.initialize()
            self.agent.train = common.function(self.agent.train)
        else:
            self.agent = self.env.get_agent()

    def build_replay_buffer(self):
        replay_buffer_capacity = 500
        if self.env.get_replay_buffer() is None:
            self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                data_spec=self.agent.collect_data_spec,
                batch_size=1,
                max_length=replay_buffer_capacity
            )
        else:
            self.replay_buffer = self.env.get_replay_buffer()

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def time_step_spec(self):
        return ts.time_step_spec(self.observation_spec())

    def start(self):
        self.log("start -> info", doprint=False)
        self.last_date = self.datas[0].datetime.date(0).strftime("%Y-%m-%d")
        self.first_date = self.datas[0].datetime.date(self.min_time_step).strftime("%Y-%m-%d")
        self.agent.train_step_counter.assign(0)
        self.log("start -> last_date: " + self.last_date, doprint=False)
        self.log("start -> first_date: " + self.first_date, doprint=False)

    def prenext_open(self):
        self.log("prenext_open -> info", doprint=True)

    def next_open(self):
        self.log("next_open -> info", doprint=True)

    def nextstart_open(self):
        self.log("nextstart_open -> info", doprint=True)

    def close(self, data=None, size=None, **kwargs):
        self.log("close -> info", doprint=True)

    def stop(self):
        self.log("stop -> info", doprint=False)

        if type(self.agent) == reinforce_agent.ReinforceAgent:
            experience = self.replay_buffer.gather_all()
        else:
            dataset = self.replay_buffer.as_dataset(num_parallel_calls=3,
                                                    sample_batch_size=64,
                                                    num_steps=2).prefetch(3)
            iterator = iter(dataset)
            experience, unused_info = next(iterator)

        train_loss = self.agent.train(experience)
        self.replay_buffer.clear()
        self.log("stop -> loss " + str(train_loss.loss.numpy()), doprint=True)
        self.log("stop -> save agent and replay buffer to env", doprint=False)
        self.env.set_agent(self.agent)
        self.env.set_replay_buffer(self.replay_buffer)

    def notify_order(self, order):
        self.log("notify_order " + str(order.status), doprint=False)
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    "BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f" %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm),
                    doprint=False)

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log("SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f" %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm),
                         doprint=False)

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled/Margin/Rejected", doprint=True)

        # Write down: no pending order
        self.order = None

    def _get_reward(self):
        reward = 0.0
        current_value = round(self.broker.getvalue(), 2)

        if self.value_queue.qsize() < 2:
            last_value = 0.0
        else:
            last_value = self.value_queue.get()

        self.log("_get_reward -> last_value: " + str(last_value), doprint=False)
        self.log("_get_reward -> current_value: " + str(current_value), doprint=False)

        if last_value != 0.0:
            reward = round((current_value - last_value) / last_value * 100, 2)

        self.log("_get_reward -> reward: " + str(reward), doprint=False)
        return reward

    def get_current_time_step(self, date):
        obv = []
        discount = [1.0]
        reward = [self._get_reward()]

        for line in self.feature_columns:
            obv.append(line[0])
        obv_len = len(obv)

        obv = tf.convert_to_tensor(np.array(obv, dtype=np.float32).reshape((-1, obv_len)))

        if date == self.first_date:
            self._current_time_step = ts.restart(obv)
        elif date == self.last_date:
            self._current_time_step = ts.termination(obv, reward)
        else:
            self._current_time_step = ts.transition(obv, reward, discount)

        self.log("get_current_time_step -> current_time_step: ", doprint=False)
        self.log(self._current_time_step, doprint=False)

        return self._current_time_step

    # def get_current_time_step(self):
    #     return self._current_time_step

    # As bt_ext.render(), but need to get observation and reward
    def next(self):
        # Simply log the closing price of the series from the reference
        self.log("next -> Close, %.2f" % self.dataclose[0], doprint=False)
        self.log("next -> Broker value: " + str(round(self.broker.getvalue(), 2)) + ", cash: " + str(
            round(self.broker.get_cash(), 2)), doprint=False)
        self.log("next -> Position size: " + str(self.position.size) + ", price: " + str(round(self.position.price, 2)),
                 doprint=False)
        self.log("next -> Fund value: " + str(self.broker.get_fundvalue()), doprint=False)
        self.date = self.datas[0].datetime.date(0).strftime("%Y-%m-%d")

        # Skip for time steps
        if self.time_step < self.min_time_step:
            self.time_step = self.time_step + 1
            self.log("next -> Skip for processing", doprint=False)
            return
        elif self.time_step == self.min_time_step:
            self.time_step = self.time_step + 1
            self.log("next -> last time step", doprint=False)
            self._current_time_step = self.get_current_time_step(self.date)
            self.log("next -> self._current_time_step", doprint=False)
            self.log(self._current_time_step, doprint=False)
            return

        self.value_queue.put(round(self.broker.getvalue(), 2), False)

        self._last_time_step = self._current_time_step
        self._last_action_step = self._action_step

        self._current_time_step = self.get_current_time_step(self.date)
        self.log("next -> self._current_time_step", doprint=False)
        self.log(self._current_time_step, doprint=False)

        self._action_step = self.agent.collect_policy.action(self._current_time_step)

        if self._last_action_step is not None and self._current_time_step is not None and self._last_time_step is not None:
            self.log("next -> self.replay_buffer.add batch", doprint=False)
            self.log("next -> self._last_time_step", doprint=False)
            self.log(self._last_time_step, doprint=False)
            self.log("next -> self._last_action_step", doprint=False)
            self.log(self._last_action_step, doprint=False)

            traj = trajectory.from_transition(self._last_time_step,
                                              self._last_action_step,
                                              self._current_time_step)
            self.replay_buffer.add_batch(traj)

        self.log("next -> self._action_step " + str(self._action_step), doprint=False)

        if self._action_step.action == 0:
            pass
        elif self._action_step.action == 1:
            # BUY, BUY, BUY!!! (with all possible default parameters)
            self.log("BUY CREATE, %.2f" % self.dataclose[0], doprint=False)
            # Keep track of the created order to avoid a 2nd order
            self.order = self.buy()
        elif self._action_step.action == 2 and self.position:
            # We must in the market before we can sell
            # SELL, SELL, SELL!!! (with all possible default parameters)
            self.log("SELL CREATE, %.2f" % self.dataclose[0], doprint=False)
            # Keep track of the created order to avoid a 2nd order
            self.order = self.sell()
