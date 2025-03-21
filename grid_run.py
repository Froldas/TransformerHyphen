import ntpath
import os
import shutil
import subprocess
import sys
import yaml


from pathlib import Path
from src.utils import load_yaml_conf

# ["SimpleTransformer", "SimpleMLP"]
MODELS = ["SimpleTransformer", "SimpleMLP"]
# ["binary", "one_hot", "simple_float", "advanced_float"]
ENCODINGS = ["binary",  "advanced_float", "one_hot"]
# ["datasets/cs-ujc.wlh", "datasets/cs-all-cstenten.wlh"]
DATASETS = ["datasets/cs-ujc.wlh"]
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
    subprocess.check_call(f"{sys.executable} train.py {cfg_name}", stderr=subprocess.DEVNULL)
    subprocess.check_call(f"{sys.executable} evaluate.py {cfg_name}", stderr=subprocess.DEVNULL)


def main():
    config = load_yaml_conf(Path(YML_CONF_PATH))

    if os.path.isdir(WORK_DIR):
        shutil.rmtree(WORK_DIR, ignore_errors=True)
    os.makedirs(WORK_DIR, exist_ok=True)

    run_index = 0
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
