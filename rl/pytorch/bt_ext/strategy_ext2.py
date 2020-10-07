# -* coding: UTF-8 -*-
from queue import Queue
import logging
import backtrader as bt
import numpy as np
import torch
from logging.config import fileConfig
from rl.pytorch.agents.DQN.dqn import DQNAgent
from rl.pytorch.agents.replay_buffer import ReplayBuffer
from rl.pytorch.networks.qnetwork import QNetwork

fileConfig("./config_file/logging_config.ini")
logger = logging.getLogger("sLogger")


class RLCommonStrategy(bt.Strategy):
    def __init__(self):
        self.train_interval = 50
        self.update_interval = 100
        logger.debug("init strategy")
        self.min_time_step = 5
        self.time_step = 1
        self.dataclose = self.datas[0].close
        self.p_change = self.datas[0].p_change

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

        self.scaled_feature_columns = []
        scaled_feature_columns = self.datas[1].columns[1:-1]
        for feature in scaled_feature_columns:
            self.scaled_feature_columns.append(getattr(self.datas[1], feature.lower()))

        self.value_queue = Queue(maxsize=2)
        self.n_actions = 3

        self.current_state = None
        self.last_state = None
        self.action = 0
        self.last_action = 0

        self.reward = 0.0

        self.q_network = None
        self.agent = None
        self.replay_buffer = None

        self.build_replay_buffer()
        self.build_agent()

    def build_agent(self):
        if self.env.get_agent() is None:
            self.q_network = QNetwork(len(self.feature_columns), None, self.n_actions)
            self.agent = DQNAgent(self.n_actions, self.q_network, self.replay_buffer)
        else:
            self.agent = self.env.get_agent()

    def build_replay_buffer(self):
        replay_buffer_capacity = 500
        if self.env.get_replay_buffer() is None:
            self.replay_buffer = ReplayBuffer(replay_buffer_capacity)
        else:
            self.replay_buffer = self.env.get_replay_buffer()

    def start(self):
        self.last_date = self.datas[0].datetime.date(0).strftime("%Y-%m-%d")
        self.first_date = self.datas[0].datetime.date(self.min_time_step).strftime("%Y-%m-%d")
        logger.debug("Start -> last date: " + self.last_date)
        logger.debug("Start -> first date: " + self.first_date)

    def prenext_open(self):
        logger.debug("prenext_open -> info")

    def next_open(self):
        logger.debug("next_open -> info")

    def nextstart_open(self):
        logger.debug("nextstart_open -> info")

    def close(self, data=None, size=None, **kwargs):
        logger.debug("close -> info")

    def stop(self):
        logger.debug("stop -> info")
        self.env.set_agent(self.agent)
        self.env.set_replay_buffer(self.replay_buffer)

    def notify_order(self, order):
        logger.debug("notify_order " + str(order.status))
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                logger.debug(
                    "BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f" %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                logger.debug(
                    "SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f" %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logger.debug("Order Canceled/Margin/Rejected")

        # Write down: no pending order
        self.order = None

    def _get_reward(self):
        reward = 0.0
        current_value = round(self.broker.getvalue(), 2)

        if self.value_queue.qsize() < 2:
            last_value = 0.0
        else:
            last_value = self.value_queue.get()

        logger.debug("_get_reward -> last_value: " + str(last_value))
        logger.debug("_get_reward -> current_value: " + str(current_value))

        if last_value != 0.0:
            reward = round((current_value - last_value) / last_value * 100, 2)
        logger.debug("_get_reward -> reward: " + str(reward))
        return reward

    def get_state(self, date):
        obv = []
        reward = [self._get_reward()]

        for line in self.scaled_feature_columns:
            obv.append(line[0])

        obv = torch.from_numpy(np.array([obv]))
        obv = torch.tensor(obv, dtype=torch.float32)
        reward = torch.from_numpy(np.array(reward))
        reward = torch.tensor(reward, dtype=torch.float32)
        return obv, reward

    # As bt_ext.render(), but need to get observation and reward
    def next(self):
        # Simply log the closing price of the series from the reference
        logger.debug("next -> Close, %.2f" % self.dataclose[0])
        logger.debug("next -> Broker value: " + str(round(self.broker.getvalue(), 2))
                     + ", cash: " + str(round(self.broker.get_cash(), 2)))
        logger.debug("next -> Position size: " + str(self.position.size)
                     + ", price: " + str(round(self.position.price, 2)))
        self.date = self.datas[0].datetime.date(0).strftime("%Y-%m-%d")

        # Skip for time steps
        if self.time_step < self.min_time_step:
            self.time_step = self.time_step + 1
            logger.debug("next -> Skip for processing")
            return
        elif self.time_step == self.min_time_step:
            self.time_step = self.time_step + 1
            logger.debug("next -> last time step")
            return

        self.last_state = self.current_state
        self.last_action = self.action

        self.current_state, self.reward = self.get_state(self.date)
        self.value_queue.put(round(self.broker.getvalue(), 2), False)
        self.action = self.agent.select_action(self.current_state)

        if self.current_state is not None and self.last_state is not None and self.last_action is not None:
            self.agent.memory.push(self.last_state, self.last_action, self.current_state, self.reward)

        if self.action.item() == 0:
            pass
        elif self.action.item() == 1:
            # BUY, BUY, BUY!!! (with all possible default parameters)
            logger.debug("BUY CREATE, %.2f" % self.dataclose[0])
            # Keep track of the created order to avoid a 2nd order
            self.order = self.buy()

        elif self.action.item() == 2 and self.position:
            # We must in the market before we can sell
            # SELL, SELL, SELL!!! (with all possible default parameters)
            logger.debug("SELL CREATE, %.2f" % self.dataclose[0])
            # Keep track of the created order to avoid a 2nd order
            self.order = self.sell()

        if self.agent.steps_done % self.train_interval == 0:
            self.agent.optimize_model()

        if self.agent.steps_done % self.update_interval == 0:
            self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())
