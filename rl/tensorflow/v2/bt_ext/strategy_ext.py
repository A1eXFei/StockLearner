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

        self.current_value = 0.0
        self.reward = 0.0
        self.discount = 1.0
        self.done = False
        self.last_observation = []
        self.current_observation = []
        self.next_observation = None

        # self.agent = self.env.getagent()
        self.episode_return = 0.0
        self.total_return = 0.0
        self.learning_rate = 1e-3
        self.train_step_counter = tf.compat.v2.Variable(0)
        self.actor_net = None
        self.agent = None
        self.replay_buffer = None
        self.time_step_spec = None
        self.action_step_spec = None
        self.last_time_step_spec = None
        self.last_action_step_spec = None

        self.feature_columns = []
        feature_columns = self.datas[0].columns[1:-1]
        for feature in feature_columns:
            self.feature_columns.append(getattr(self.datas[0], feature.lower()))
        self.action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
        self.observation_spec = array_spec.ArraySpec(shape=(len(feature_columns),), dtype=np.float, name='observation')

        self.action_spec = tensor_spec.from_spec(self.action_spec)
        self.observation_spec = tensor_spec.from_spec(self.observation_spec)

    def start(self):
        self.log("start -> info", doprint=True)
        self.last_date = self.datas[0].datetime.date(0).strftime("%Y-%m-%d")
        self.episode_return = 0

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
        self.time_step_spec = ts.termination(np.array(self.current_observation, dtype=np.float), self.reward)
        self.episode_counter += 1
        experience = self.replay_buffer.gather_all()
        train_loss = self.tf_agent.train(experience)
        self.replay_buffer.clear()

        step = self.agent.train_step_counter.numpy()
        print('step = {0}: loss = {1}'.format(step, train_loss.loss))

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
        self.log("_get_obv", doprint=True)

        for line in self.feature_columns:
            obv.append(line[0])

        if date == self.last_date:
            done = True

        return obv, done

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
            self.current_observation, self.done = self._get_observation(date=self.date)
            self.time_step_spec = ts.restart(np.array(self.current_observation, dtype=np.float))

            self.log("next -> Last time step", doprint=True)
            self.log(self.time_step_spec, doprint=True)
            self.log(self.observation_spec, doprint=True)
            self.log(self.action_spec, doprint=True)

            self.log(type(self.observation_spec), doprint=True)
            self.log(type(self.action_spec), doprint=True)

            fc_layer_params = (100,)
            batch_size = 32
            replay_buffer_capacity = 100

            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)

            self.actor_net = actor_distribution_network.ActorDistributionNetwork(
                self.observation_spec,
                self.action_spec,
                fc_layer_params=fc_layer_params)

            self.agent = reinforce_agent.ReinforceAgent(self.time_step_spec,
                                                        self.action_spec,
                                                        actor_network=self.actor_net,
                                                        optimizer=optimizer,
                                                        normalize_returns=True,
                                                        train_step_counter=self.train_step_counter)

            self.agent.initialize()
            self.agent.train = common.function(self.agent.train)
            self.log("!!!!!!!!!!!!!!!!!!!!!!!!!!", doprint=True)
            self.log(self.agent.collect_data_spec, doprint=True)
            self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                data_spec=self.agent.collect_data_spec,
                batch_size=batch_size,
                max_length=replay_buffer_capacity)

            return

        # print("date in next()")
        # print(self.date)

        # Fetch observation, do the rest thing first which should be done in step function
        self.last_observation = self.current_observation
        self.last_time_step_spec = self.time_step_spec
        self.last_action_step_spec = self.action_step_spec

        self.current_value = round(self.broker.getvalue(), 2)
        self.current_observation, self.done = self._get_observation(date=self.date)
        self.reward = self._get_reward()

        self.log("next -> last obv", doprint=True)
        self.log(self.last_observation, doprint=True)
        self.log("next -> curr obv", doprint=True)
        self.log(self.current_observation, doprint=True)

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
