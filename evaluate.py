import numpy as np
import torch

from sklearn.metrics import accuracy_score
from torch import load
from pathlib import Path

from src.dataset import HyphenationDataset, HyphenationInterace, insert_hyphenation
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

    dataset = HyphenationDataset(data_file=config["dataset"],
                                 work_dir=config["work_dir"],
                                 print_info=config["print_dataset_statistics"])
    X = []
    y = []
    for data_point in dataset:
        features, label = data_point
        X.append(features)  # Convert features to NumPy array
        y.append(label)

    y_pred = loaded_model(torch.Tensor(np.array(X)).to("cpu"))

    accuracy = accuracy_score(torch.Tensor(np.array(y)).detach().numpy(), y_pred.to("cpu").detach().numpy())

    print(f"Accuracy: {accuracy}")
    with open(Path(config["work_dir"]) / config["mispredict_path"], "w+", encoding="utf-8") as f:
        for i in range(len(dataset)):
            if not torch.equal(y_pred[i], torch.Tensor(y[i])):
                f.writelines(f"GT: {dataset.words[i]} | PRED: {insert_hyphenation(dataset.words[i].replace("-", ""), y_pred[i])}\n")


if __name__ == "__main__":
    main()
