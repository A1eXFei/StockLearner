import random
from os.path import join

from feed.bt_data import *
from rl.common_bt_ext.cerebro_ext import RLCerebro
from rl.common_bt_ext.sizer_ext import PercentSizer
from rl.pytorch.bt_ext.strategy_ext2_pytorch import RLCommonStrategy

# from rl.v2.bt_ext.strategy_ext2_pytorch import RLCommonStrategy

data_path = "./test_data/stock/tech"
data_files = "./test_data/stock/tech/000002.csv"
scaled_data_path = "./test_data/stock/tech"
scaled_data_files = "./test_data/stock/tech/000002_s.csv"
data_schema = "./config_file/schema/tech_data_schema.yaml"


def get_data_files(file_list):
    index = random.randint(0, len(file_list) - 1)
    csv_file = join(data_path, file_list[index])
    scaled_csv_file = join(data_path, file_list[index].replace(".csv", "_s.csv"))
    return csv_file, scaled_csv_file


if __name__ == "__main__":
    # data_files = [f for f in listdir(data_path) if f != ".DS_Store" and "_s" not in f]
    agent = None
    replay_buffer = None
    iterations = 100000
    print(data_files)

    for i in range(iterations):
        print("Iterator for " + str(i))
        cerebro = RLCerebro()

        # Add a strategy
        cerebro.addstrategy(RLCommonStrategy)

        if agent is not None and replay_buffer is not None:
            cerebro.set_agent(agent)
            cerebro.set_replay_buffer(replay_buffer)

        data = BTCSVTechData(
            dataname=data_files,
            reverse=False
        )

        scaled_data = BTCSVTechData(
            dataname=scaled_data_files,
            reverse=False
        )

        # Add data feed to cerebo
        cerebro.adddata(data)
        cerebro.adddata(scaled_data, name="scaled_data")

        # Add sizer
        cerebro.addsizer(PercentSizer)

        # Set the commission - 0.1% ... divide by 100 to remove the %
        cerebro.broker.setcommission(commission=0.001)

        # Set our desired cash start
        cerebro.broker.set_cash(100000.0)
        print("Starting portfolio value: %.2f" % cerebro.broker.getvalue())

        cerebro.run()

        final_portfolio = round(cerebro.broker.getvalue(), 2)
        print("Final portfolio value: %.2f" % cerebro.broker.getvalue())

        # cerebro.plot()
        agent = cerebro.get_agent()
        replay_buffer = cerebro.get_replay_buffer()
