# -* coding: UTF-8 -*-


class BaseAgent:
    def __init__(self):
        pass

    def select_action(self, state):
        raise NotImplementedError

    def optimize_model(self):
        raise NotImplementedError
