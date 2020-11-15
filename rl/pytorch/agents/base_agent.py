# -* coding: UTF-8 -*-
import torch
import logging
from abc import ABCMeta
from abc import abstractmethod
from logging.config import fileConfig


class BaseAgent(metaclass=ABCMeta):
    def __init__(self, n_actions, batch_size=128):
        fileConfig("./config_file/logging_config.ini")
        self.logger = logging.getLogger("sLogger")
        self.n_actions = n_actions
        self.batch_size = batch_size
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

    @abstractmethod
    def select_action(self, state):
        raise NotImplementedError

    @abstractmethod
    def optimize_model(self):
        raise NotImplementedError

    def take_optimisation_step(self, optimizer, loss, network, clipping_norm=None, retain_graph=False):
        if not isinstance(network, list):
            network = [network]
        optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        self.logger.debug("Loss -- {}".format(loss.item()))
        if clipping_norm is not None:
            for net in network:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clipping_norm)
        optimizer.step()  # this applies the gradients
