from model import MLP


def get_model(model_type, model_config_path, model_name):
    if model_type == "MLP":
        model = MLP.MLP(config_file=model_config_path, model_name= model_name)
        return model

    raise ModelTypeNotFound()


class ModelTypeNotFound(Exception):
    def __init__(self):
        print("Model Type is not found")