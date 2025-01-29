import os
import yaml
import importlib


class Config(dict):
    """Subclass of dict for accessing keys with '.' like class attributes."""

    def __init__(self, path, save=None):
        assert isinstance(path, str) or isinstance(
            path, dict
        ), f"Path should be either dict or path to yaml file, fond {type(path)}."
        if isinstance(path, str):
            assert os.path.exists(path), f"Path {path} does not exist."
            path = yaml.safe_load(open(path, "r"))
        super().__init__(path)

    def __getattr__(self, name):
        try:
            value = self[name]
            if isinstance(value, dict):
                value = Config(value)
            return value
        except KeyError:
            raise AttributeError(name)


def config_to_instance(**config):
    """
    Instantites the classr or method config[name].
    For a class, arguments are specified as additional entries in the dict.

    Args:
        config: dict cainting the name of the class or method to create

    """
    module, attr = os.path.splitext(config.pop("name"))
    module = importlib.import_module(module)
    attr = getattr(module, attr[1:])
    if config:
        attr = attr(**config)
    return attr
