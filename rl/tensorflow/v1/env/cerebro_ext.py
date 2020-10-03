import backtrader as bt
import pandas as pd


class RLExtCerebro(bt.Cerebro):
    BASIC_COLUMNS = ["DATE", "OPEN", "HIGH", "CLOSE", "LOW", "VOLUME", "PRICE_CHANGE", "P_CHANGE", "TURNOVER",
                          "LABEL"]

    def __init__(self):
        bt.Cerebro.__init__(self)

        self._agent = None
        self._tf_sess = None
        self._df = None
        self._scaled_df = None
        self._global_step = 0

    def addagent(self, agent):
        self._agent = agent

    def getagent(self):
        return self._agent

    def getdf(self):
        return self._df, self._scaled_df

    def run(self):
        if self._agent is None or self._df is None:
            raise NotImplementedError

        bt.Cerebro.run(self)

    def adddf(self, data_path, columns=BASIC_COLUMNS):
        self._df = self._load_pd(data_path, columns)

    def addscaleddf(self, data_path, columns=BASIC_COLUMNS):
        self._scaled_df = self._load_pd(data_path, columns)

    def addglobalstep(self, step):
        self._global_step = step

    def getglobalstep(self):
        return self._global_step

    def _load_pd(self, data_path, columns=BASIC_COLUMNS):
        df = pd.read_csv(data_path, header=None)
        df.columns = columns
        df.set_index("DATE", inplace=True)

        if "LABEL" in columns:
            df.drop(["LABEL"], axis=1, inplace=True)

        return df
