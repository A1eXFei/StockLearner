from __future__ import division


class DQNAgent:
    def choose_action(self, observation):
        return self.Hold

    def backward(self, reward, terminal):
        pass

    def compile(self, optimizer, metrics=[]):
        pass

    def load_weights(self, filepath):
        pass

    def save_weights(self, filepath, overwrite=False):
        pass

    def __init__(self):
        super(DQNAgent, self).__init__()

        self.Hold, self.Buy, self.Sell = range(3)
        self.actions = [self.Hold, self.Buy, self.Sell]
