import os
import yaml


def load_yaml_conf(path: str | os.PathLike):
    with open(path) as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)