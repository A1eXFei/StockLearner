import random
from os import listdir
from os.path import join

import yaml

from feed.bt_data import BTCSVBasicData
from rl.tensorflow.v1.agent import DQNAgent
from rl.tensorflow.v1.env import PercentSizer
from rl.tensorflow.v1.env.cerebro_ext import RLExtCerebro

APP_CONFIG_FILE_PATH = "./config_file/train/tf_keras_rl_ops.yaml"

yaml_file = open(APP_CONFIG_FILE_PATH, 'r', encoding='utf-8')
yaml_config = yaml.load(yaml_file.read())

data_path = yaml_config["dataset"]["data_path"]
# TODO: Tech schema columns
schema_path = yaml_config["dataset"]["schema_path"]
episode = yaml_config["run"]["episode"]


def get_data_files(file_list):
    index = random.randint(0, len(file_list) - 1)
    csv_file = join(data_path, file_list[index])
    scaled_csv_file = join(data_path, file_list[index].replace(".csv", "_s.csv"))
    return csv_file, scaled_csv_file


if __name__ == "__main__":
    data_files = [f for f in listdir(data_path) if f != ".DS_Store" and "_s" not in f]

    agent = DQNAgent(yaml_config)
    global_step = 0

    for i in range(episode):
        print("#####################EPISODE " + str(i) + "###########################")

        curr_data_path, curr_scaled_data_path = get_data_files(data_files)
        print("Current file is: " + data_path)
        print("Related scaled file is: " + curr_scaled_data_path)

        cerebro = RLExtCerebro()

        # Add a agent
        cerebro.addagent(agent)

        # Add a strategy
        cerebro.addstrategy(RLCommonStrategy)

        # Create a Data Feed
        data = BTCSVBasicData(
            dataname=curr_data_path,
            reverse=False)

        # Add the Data Feed to Cerebro
        cerebro.adddata(data)

        # Add additional schema frame
        cerebro.adddf(curr_data_path, columns=RLExtCerebro.BASIC_COLUMNS)
        cerebro.addscaleddf(curr_scaled_data_path, columns=RLExtCerebro.BASIC_COLUMNS)
        cerebro.addglobalstep(global_step)

        # Add sharpe ratio analyzer to Cerebro
        # cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='mysharpe')

        # Add a FixedSize sizer according to the stake
        # cerebro.addsizer(bt.sizers.FixedSize, stake=100)
        cerebro.addsizer(PercentSizer)

        # Set the commission - 0.1% ... divide by 100 to remove the %
        cerebro.broker.setcommission(commission=0.001)

        # Set our desired cash start
        cerebro.broker.setcash(100000.0)
        print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())

        cerebro.run()

        final_portfolio = round(cerebro.broker.getvalue(), 2)
        print("Final Portfolio Value: " + str(final_portfolio))
        # thestrat = thestrats[0]
        # print('Sharpe Ratio:', thestrat.analyzers.mysharpe.get_analysis())
        assert final_portfolio >= 0, "????"

        # cerebro.plot()
        global_step = cerebro.getglobalstep()
        # time.sleep(5)
        # print(global_step)

