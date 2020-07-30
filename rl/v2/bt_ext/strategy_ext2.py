import backtrader as bt
import numpy as np
import tensorflow as tf
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.networks import actor_distribution_network
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
        self.min_time_step = 5
        self.time_step = 1
        self.dataclose = self.datas[0].close
        self.p_change = self.datas[0].p_change

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.date = None
        self.last_date = None

        self._observation = []
        self._episode_ended = False

        self.last_action = None
        self.action = None
        self.last_observation = None
        self.observation = None
        self.done = False

        self.reward = 0.0
        self.agent = None
        self.actor_net = None
        self.replay_buffer = None
        self.learning_rate = 1e-3
        self.train_step_counter = tf.compat.v2.Variable(0)

        self.feature_columns = []
        feature_columns = self.datas[0].columns[1:-1]
        for feature in feature_columns:
            self.feature_columns.append(getattr(self.datas[0], feature.lower()))

        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
        self._observation_spec = array_spec.ArraySpec(shape=(len(feature_columns),), dtype=np.float, name='observation')
        self._action_spec = tensor_spec.from_spec(self._action_spec)
        self._observation = tensor_spec.from_spec(self._observation_spec)

        batch_size = 32
        replay_buffer_capacity = 100
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.actor_net = actor_distribution_network.ActorDistributionNetwork(
            self.observation_spec(),
            self.action_spec()
        )

        self.agent = reinforce_agent.ReinforceAgent(
            self.time_step_spec(),
            self.action_spec(),
            actor_network=self.actor_net,
            optimizer=optimizer,
            normalize_returns=True,
            train_step_counter=self.train_step_counter
        )

        self.agent.initialize()
        self.agent.train = common.function(self.agent.train)
        self.log("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", doprint=True)
        self.log(self.agent.collect_data_spec, doprint=True)
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=batch_size,
            max_length=replay_buffer_capacity
        )

        self._current_time_step = None
        self._last_time_step = None
        self._next_time_step = None
        self._action_step = None
        self._last_action_step = None
        self._next_action_step = None

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def time_step_spec(self):
        return ts.time_step_spec(self.observation_spec())

    def start(self):
        self.log("start -> info", doprint=True)
        self.last_date = self.datas[0].datetime.date(0).strftime("%Y-%m-%d")
        self.agent.train_step_counter.assign(0)

    def prenext_open(self):
        self.log("prenext_open -> info", doprint=True)

    def next_open(self):
        self.log("next_open -> info", doprint=True)

    def nextstart_open(self):
        self.log("nextstart_open -> info", doprint=True)

    def close(self, data=None, size=None, **kwargs):
        self.log("close -> info", doprint=True)

    def stop(self):
        self.log("stop -> info", doprint=True)
        self._episode_ended = False
        ts.termination(np.array(self._observation, dtype=np.float), self.reward)

        experience = self.replay_buffer.gather_all()
        train_loss = self.agent.train(experience)
        self.replay_buffer.clear()
        self.log("stop -> loss = {1}".format(train_loss.loss), doprint=True)

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
        self.log("_get_reward", doprint=True)
        return 0.0

    def _get_observation(self, date):
        obv = []
        done = False
        self.log("_get_observation -> ", doprint=True)

        for line in self.feature_columns:
            obv.append(line[0])

        if date == self.last_date:
            done = True

        obv = np.array(obv, dtype=np.float)
        return obv, done

    def get_current_time_step(self):
        return self._current_time_step

    # As bt_ext.render(), but need to get observation and reward
    def next(self):
        # Simply log the closing price of the series from the reference
        self.log("next -> Close, %.2f" % self.dataclose[0], doprint=True)
        self.log("next -> Broker value: " + str(round(self.broker.getvalue(), 2)) + ", cash: " + str(
            round(self.broker.get_cash(), 2)), doprint=True)
        self.log("next -> Position size: " + str(self.position.size) + ", price: " + str(round(self.position.price, 2)),
                 doprint=True)
        self.date = self.datas[0].datetime.date(0).strftime("%Y-%m-%d")

        # Skip for time steps
        if self.time_step < self.min_time_step:
            self.time_step = self.time_step + 1
            self.log("next -> Skip for processing", doprint=True)
            return
        elif self.time_step == self.min_time_step:
            self.time_step = self.time_step + 1
            self.observation, self.done = self._get_observation(self.date)
            self.log("next -> last time step", doprint=True)
            return

        self._last_time_step = self._current_time_step
        self._last_action_step = self._action_step

        # TODO: complete for self_current_step values
        self._current_time_step = ()

        if self._last_action_step is not None and self._current_time_step is not None and self._last_time_step is not None:
            traj = trajectory.from_transition(self._last_time_step, self._last_action_step, self._current_time_step)
            self.replay_buffer.add_batch(traj)

        if self.done:
            self.time_step_spec = ts.termination(np.array(self.current_observation, dtype=np.float), self.reward)
        else:
            self.time_step_spec = ts.transition(np.array(self.current_observation, dtype=np.float), reward=self.reward,
                                                discount=self.discount)

        self.log("next -> time_step_spec", doprint=False)
        self.log(self.time_step_spec, doprint=False)

        if self.last_action_step_spec is not None:
            traj = trajectory.from_transition(self.last_time_step_spec, self.last_action_step_spec, self.time_step_spec)
            self.replay_buffer.add_batch(traj)

        if self.done:
            return

        # self.action = self.agent.choose_action(self.current_observation)
        # self.action = 0

        self.action_step_spec = self.agent.collect_policy.action(self.time_step_spec)

        if self.action_step_spec.action == 0:
            pass
        elif self.action_step_spec.action == 1:
            # BUY, BUY, BUY!!! (with all possible default parameters)
            self.log("BUY CREATE, %.2f" % self.dataclose[0], doprint=False)
            # Keep track of the created order to avoid a 2nd order
            self.order = self.buy()

        elif self.action_step_spec.action and self.position:
            # We must in the market before we can sell
            # SELL, SELL, SELL!!! (with all possible default parameters)
            self.log("SELL CREATE, %.2f" % self.dataclose[0], doprint=False)
            # Keep track of the created order to avoid a 2nd order
            self.order = self.sell()
