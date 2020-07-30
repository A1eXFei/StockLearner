import yaml
from backtrader.feeds import GenericCSVData


#              0       1       2       3        4      5         6               7           8           9
BT_COLUMNS = ["date", "open", "high", "close", "low", "volume", "label"]


class BTAbstractCSVData(GenericCSVData):
    @staticmethod
    def get_params(columns):
        params = [('nullvalue', float('NaN')), ('dtformat', '%Y-%m-%d'), ('tmformat', '%H:%M:%S'), ('headers', False)]

        for i in range(len(columns)):
            p = (columns[i].lower(), i)
            params.append(p)

        params.append(('time', -1))
        params.append(('openinterest', -1))
        return tuple(params)

    @staticmethod
    def get_columns(schema_path):
        columns = []
        yaml_file = open(schema_path, 'r', encoding='utf-8')
        yaml_config = yaml.load(yaml_file.read())
        for each_field in yaml_config["schema"]:
            columns.append(each_field["field_name"].lower())
        return columns

    @staticmethod
    def get_lines(schema_path):
        columns = []
        yaml_file = open(schema_path, 'r', encoding='utf-8')
        yaml_config = yaml.load(yaml_file.read())
        for each_field in yaml_config["schema"]:
            columns.append(each_field["field_name"].lower())

        for column in BT_COLUMNS:
            columns.remove(column)

        return tuple(columns)


class BTCSVBasicData(BTAbstractCSVData):
    DATA_SCHEMA = "./config_file/schema/basic_data_schema.yaml"
    columns = BTAbstractCSVData.get_columns(DATA_SCHEMA)
    lines = BTAbstractCSVData.get_lines(DATA_SCHEMA)
    params = BTAbstractCSVData.get_params(columns)


class BTCSVTechData(BTAbstractCSVData):
    DATA_SCHEMA = "./config_file/schema/tech_data_schema.yaml"
    columns = BTAbstractCSVData.get_columns(DATA_SCHEMA)
    lines = BTAbstractCSVData.get_lines(DATA_SCHEMA)
    params = BTAbstractCSVData.get_params(columns)
