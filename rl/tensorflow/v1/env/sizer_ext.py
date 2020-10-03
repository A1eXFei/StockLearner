#!/usr/bin/bt_ext python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
#
# Copyright (C) 2015, 2016 Daniel Rodriguez
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
###############################################################################
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import backtrader as bt

__all__ = ['PercentSizer', 'AllInSizer']


class PercentSizer(bt.Sizer):
    '''This sizer return percents of available cash

    Params:
      - ``percents`` (default: ``20``)
      - ``mode`` (default: ``1``)
    '''

    params = (
        ('percents', 50),
        ('mode', 1),
        ("printlog", False),
    )

    def __init__(self):
        pass

    def log(self, txt, doprint=False):
        """ Logging function fot this strategy"""
        if self.params.printlog or doprint:
            print(txt)

    def _getsizing(self, comminfo, cash, data, isbuy):
        position = self.broker.getposition(data)

        if data.close[0] == 0:
            return 0

        if self.params.mode == 1:
            if isbuy:
                size = (cash / data.close[0]) * (self.params.percents / 100) // 100 * 100
                self.log("Mode 1, Buy for " + str(size))
            else:
                if position:
                    size = position.size * (self.params.percents / 100) // 100 * 100
                    self.log("Mode 1, Sell for " + str(size))
                else:
                    size = 0

        if self.params.mode == 2:
            if not position:
                size = (cash / data.close[0]) * (self.params.percents / 100) // 100 * 100
                self.log("Mode 2, Buy for " + str(size))
            else:
                size = position.size
                self.log("Mode 2, Sell for " + str(size))
        return size


class AllInSizer(PercentSizer):
    '''This sizer return all available cash of broker

     Params:
       - ``percents`` (default: ``100``)
     '''
    params = (
        ('percents', 100),
    )
