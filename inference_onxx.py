import sys

from pathlib import Path
from torch import Tensor

from src.dataset import HyphenationInterface

import onnxruntime as ort
from src.utils import load_yaml_conf, insert_hyphenation

YML_CONF_PATH = "configuration.yml"


def main():
    config = load_yaml_conf(Path(YML_CONF_PATH))
    hyp_itf = HyphenationInterface.load_configuration(config["work_dir"], config["configuration_path"])

    session = ort.InferenceSession(Path(config["work_dir"]) / config["onxx_model_path"])


    # for word in data:
    for word in sys.argv[1:]:
        input_tensor = hyp_itf.encode(word)
        output = session.run(None, {"input": input_tensor})
        print(insert_hyphenation(word, output[0][0]))


if __name__ == "__main__":
    #from cProfile import run
    #run("main()")
    main()
