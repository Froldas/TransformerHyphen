import ntpath
import os
import shutil
import subprocess
import sys
import yaml

from pathlib import Path
from src.utils import load_yaml_conf

# ["SimpleTransformer", "SimpleMLP", "SimpleTransformerResidual", "SimpleTransformerMasked", "SimpleReverseTransformer"]
MODELS = [#"SimpleMLP",
          #"SimpleDeeperMLP",
          #"SimpleTransformer",
          #"SimpleTransformerResidual",
          #"SimpleTransformerReversed",
          "AdvancedTransformerResidualDeep"]

# ["binary", "one_hot", "simple_float", "advanced_float"]
ENCODINGS = ["advanced_binary", "simple_embedding", "one_hot"]

# ["datasets/cs-ujc.wlh", "datasets/cs-tenten.wlh", "datasets/wortliste.wlh"]
DATASETS = ["datasets/merged.wlh"]
YML_CONF_PATH = "configuration.yml"
WORK_DIR = Path("grid_work")


def create_config(config, conf_name):
    with open(conf_name, 'w+') as f:
        f.writelines("---\n")
        yaml.dump_all([config], f)


def run(config, run_index):
    cfg_name = WORK_DIR / f"temp_cfg_{run_index}.yml"
    create_config(config, cfg_name)
    print(f"Commencing training #{run_index}.")
    subprocess.check_call(f"{sys.executable} train.py {cfg_name}", shell=True)


def main():
    config = load_yaml_conf(Path(YML_CONF_PATH))

    if os.path.isdir(WORK_DIR):
        shutil.rmtree(WORK_DIR, ignore_errors=True)
    os.makedirs(WORK_DIR, exist_ok=True)

    run_index = 0

    # list(itertools.product([MODELS, ENCODINGS, DATASETS, ]])
    for model in MODELS:
        for encoding in ENCODINGS:
            for dataset in DATASETS:
                config["model"] = model
                config["encoding"] = encoding
                config["dataset"] = dataset
                dataset_name = ntpath.basename(dataset).split(".")[0]
                config["work_dir"] = str(WORK_DIR / f"build-{model}-{encoding}-{dataset_name}")
                run(config, run_index)
                run_index += 1


if __name__ == "__main__":
    main()
