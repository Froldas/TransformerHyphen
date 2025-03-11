import sys

from pathlib import Path
from torch import Tensor, load

from src.dataset import HyphenationInterace, insert_hyphenation
from src.ModelDict import ModelDict
from src.utils import load_yaml_conf

YML_CONF_PATH = "configuration.yml"


def main():
    config = load_yaml_conf(Path(YML_CONF_PATH))
    hyp_itf = HyphenationInterace.load_configuration(config["work_dir"], config["configuration_path"])

    model_path = Path(config["work_dir"]) / config["model_path"]
    loaded_model = ModelDict(hyp_itf.num_input_tokens, hyp_itf.embed_size, hyp_itf.output_size).models[config["model"]]
    loaded_model.load_state_dict(load(model_path))
    loaded_model.eval()

    #with open("datasets/test_text.txt", "r+", encoding="utf-8") as f:
    #    data = f.readline()
    #data = data.split(" ")
    # for word in data:
    for word in sys.argv[1:]:
        input_tensor = Tensor(hyp_itf.convert_word_to_input_tensor(word))
        output = loaded_model(input_tensor)
        print(insert_hyphenation(word, output[0]))


if __name__ == "__main__":
    #from cProfile import run
    #run("main()")
    main()
