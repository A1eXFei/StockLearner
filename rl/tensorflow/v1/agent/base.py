import backtrader as bt
import numpy as np


class RLBaseAgent:
    def __init__(self):
        # TODO: Change to action with sizer (1/4, 1/2, 3/4, 1)
        # Hold, BuyHalf, BuyAll, BuyOneQuarter, BuyThreeQuarter, SellHalf, SellAll, SellOneQuarter, SellThreeQuarter
        self.Hold, self.Buy, self.Sell = range(3)
        self.actions = [self.Hold, self.Buy, self.Sell]

    def random_action(self):
        idx = np.random.random_integers(0, len(self.actions) - 1)
        return self.actions[idx]

    def choose_action(self, observation):
        raise NotImplementedError

    def store(self):
        raise NotImplementedError

    def choose_action(self):
        raise NotImplementedError

    def study(self):
        raise NotImplementedError


class RLCommonStrategy(bt.Strategy):
    params = (
        ("printlog", True),
    )

    def log(self, txt, dt=None, doprint=False):
        """ Logging function fot this strategy"""
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print("%s, %s" % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the schema[0] dataseries
        self.dataclose = self.datas[0].close
        # self.p_change = self.datas[0].p_change

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.date = None

        self.last_value = self.broker.get_cash()
        self.current_value = 0.

        self.agent = self.env.getagent()
        self.df, self.scaled_df = self.env.getdf()
        self.last_observation = None
        self.current_observation = None
        self.reward = 0
        self.action = None
        self.done = False

        # self.global_step = 0
        self.learning_freq = self.agent.learning_freq

        # As bt_ext.step(), but next observation, reward and done will be given in next function.
    def notify_order(self, order):
        # self.log("notify_order " + str(order.status))
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
            self.log("Order Canceled/Margin/Rejected")

        # Write down: no pending order
        self.order = None

    def __get_reward(self, calculate_type="upl"):
        # upl = unrealized profit or loss, pct = percentage of win or loss
        reward = 0.
        # self.log("Last value " + str(self.last_value) + " current value: " + str(self.current_value))
        if calculate_type == "upl":
            reward = self.current_value - self.last_value

        if calculate_type == "pct":
            reward = round((self.current_value - self.last_value) / self.last_value, 10)

        # self.log("Reward: " + str(reward))
        return reward

    def __get_observation(self, date, offset=0, scaled_data=False):
        def get_data(df):
            max_idx = df.shape[0]
            # print(max_idx)
            idx = df.index.get_loc(date)
            # print("index in _get_observation")
            # print(idx)
            if offset > 0:
                raise IndexError
            elif offset < 0:
                offset + 1
                df2 = df.iloc[idx + offset:idx + 1]
            else:
                df2 = df.loc[date]
            done = False
            if idx == max_idx:
                done = True
            return df2.values.tolist(), done

        if scaled_data:
            if self.scaled_df is None:
                raise NotImplementedError
            else:
                return get_data(self.scaled_df)
        else:
            return get_data(self.df)

    # As bt_ext.render(), but need to get observation and reward
    def next(self):
        # Simply log the closing price of the series from the reference
        # self.log("Next Close, %.2f" % self.dataclose[0])
        self.log("Broker value: " + str(round(self.broker.getvalue(), 2)) + ", cash: " + str(round(self.broker.get_cash(), 2)))
        self.log("Position size: " + str(self.position.size) + ", price: " + str(round(self.position.price, 2)))

        self.date = self.datas[0].datetime.date(0).strftime("%Y-%m-%d")
        # print("date in next()")
        # print(self.date)

        # Fetch observation, do the rest thing first which should be done in step function
        self.last_observation = self.current_observation
        self.current_value = round(self.broker.getvalue(), 2)
        self.current_observation, self.done = self.__get_observation(date=self.date, scaled_data=True)
        self.reward = self.__get_reward(calculate_type="pct")
        self.agent.store(self.last_observation, self.action, self.reward, self.current_observation, self.done)

        self.action = self.agent.choose_action(self.current_observation)

        if self.action == self.agent.Hold:
            pass
        elif self.action == self.agent.Buy:
            # BUY, BUY, BUY!!! (with all possible default parameters)
            self.log("BUY CREATE, %.2f" % self.dataclose[0], doprint=False)

            # Keep track of the created order to avoid a 2nd order
            self.order = self.buy()

        elif self.action == self.agent.Sell and self.position:
            # We must in the market before we can sell
            # SELL, SELL, SELL!!! (with all possible default parameters)
            self.log("SELL CREATE, %.2f" % self.dataclose[0], doprint=False)

            # Keep track of the created order to avoid a 2nd order
            self.order = self.sell()

        global_step = self.env.getglobalstep()

        self.agent.study(global_step)

        self.env.addglobalstep(global_step + 1)

