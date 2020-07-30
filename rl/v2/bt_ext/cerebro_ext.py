import backtrader as bt


class RLCerebro(bt.Cerebro):
    BASIC_COLUMNS = ["DATE", "OPEN", "HIGH", "CLOSE", "LOW", "VOLUME", "PRICE_CHANGE", "P_CHANGE", "TURNOVER",
                     "LABEL"]

    def __init__(self):
        super(RLCerebro, self).__init__()
        self._agent = None
        self._replay_buffer = None

    def get_replay_buffer(self):
        return self._replay_buffer

    def set_replay_buffer(self, replay_buffer):
        self._replay_buffer = replay_buffer

    def get_agent(self):
        return self._agent

    def set_agent(self, agent):
        self._agent = agent

    def run(self):
        bt.Cerebro.run(self)
