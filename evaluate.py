import numpy as np
import torch
import sys

from sklearn.metrics import accuracy_score, recall_score
from torch import load
from pathlib import Path

from src.dataset import HyphenationDataset, HyphenationInterface
from src.ConfDict import Models, Encodings
from src.utils import load_yaml_conf, insert_hyphenation


YML_CONF_PATH = "configuration.yml"


def main():
    if len(sys.argv) > 1:
        config = load_yaml_conf(Path(sys.argv[1]))
    else:
        config = load_yaml_conf(Path(YML_CONF_PATH))

    hyp_itf = HyphenationInterface.load_configuration(config["work_dir"], config["configuration_path"])
    model_path = Path(config["work_dir"]) / config["model_path"]
    loaded_model = Models(hyp_itf.num_input_tokens, hyp_itf.encoding_size, hyp_itf.output_size).models[config["model"]]
    loaded_model.load_state_dict(load(model_path))
    loaded_model.eval()

    dataset = HyphenationDataset(data_file=config["dataset"],
                                 work_dir=config["work_dir"],
                                 encoding=Encodings().encodings[config["encoding"]],
                                 print_info=config["print_dataset_statistics"])
    X = []
    y = []
    for data_point in dataset:
        features, label = data_point
        X.append(features)  # Convert features to NumPy array
        y.append(label)

    x_pred = loaded_model(torch.Tensor(np.array(X)).to("cpu"))

    accuracy = accuracy_score(torch.Tensor(np.array(y)).detach().numpy(), x_pred.to("cpu").detach().numpy())
    recall = recall_score(torch.Tensor(np.array(y)).detach().numpy(), x_pred.to("cpu").detach().numpy(),average="samples")

    with open(Path(config["work_dir"]) / "eval_metrics.log", "w+", encoding="utf-8") as f:
        f.writelines(f"Accuracy: {accuracy:.4f}\n")
        print(f"Accuracy: {accuracy:.4f}")
        f.writelines(f"Recall: {recall:.4f}\n")
        print(f"Recall: {recall:.4f}")

    if not config["fast_eval"]:
        with open(Path(config["work_dir"]) / config["mispredict_path"], "w+", encoding="utf-8") as f:
            for i in range(len(dataset)):
                if not torch.equal(x_pred[i], torch.Tensor(y[i])):
                    f.writelines(f"GT: {dataset.words[i]} | PRED: {insert_hyphenation(dataset.words[i].replace('-', ''), x_pred[i])}\n")


if __name__ == "__main__":
    main()
