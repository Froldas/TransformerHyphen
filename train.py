import logging
import os
import shutil
import sys
from pathlib import Path

import torch.nn as nn
import torch.optim as optim

import src.utils as utils
from src.ConfDict import Models, Encodings
from src.dataset import HyphenationDataset, HyphenationDatasetSlidingWindow
from src.patgen import train_patgen, eval_patgen

YML_CONF_PATH = "configuration.yml"


def main():
    if len(sys.argv) > 1:
        config = utils.load_yaml_conf(Path(sys.argv[1]))
    else:
        config = utils.load_yaml_conf(Path(YML_CONF_PATH))

    os.makedirs(config["work_dir"], exist_ok=True)

    utils.setup_logger(Path(config["work_dir"]) / config["training_log_path"])
    # Check if CUDA is available
    device = "cpu"  #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    # set seed for reproducibility
    utils.set_seed(config["seed"])

    # Create datasets and dataloaders
    if config["sliding_window"]:
        dataset = HyphenationDatasetSlidingWindow(data_file=config["dataset"],
                                                  work_dir=config["work_dir"],
                                                  encoding=Encodings().encodings[config["encoding"]],
                                                  print_info=config["print_dataset_statistics"])
    else:
        dataset = HyphenationDataset(data_file=config["dataset"],
                                     work_dir=config["work_dir"],
                                     encoding=Encodings().encodings[config["encoding"]],
                                     print_info=config["print_dataset_statistics"])

    # note: patgen requires dumping the datasets
    train_dataset, test_dataset = utils.split_dataset(dataset, config["train_split"],
                                                      work_dir=config["work_dir"],
                                                      dump_datasets=config["patgen"])

    model = Models(dataset.num_input_tokens,
                   dataset.encoding_size,
                   dataset.output_size).models[config["model"]].to(device)

    loss_func = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.05)

    # Training
    utils.model_training(model, train_dataset, config["num_epochs"], optimizer, loss_func, config["batch_size"], device)

    # Dump trained model
    quantized_model = utils.quantize_model(model)
    utils.save_model(model, Path(config["work_dir"]) / config["model_path"])
    utils.save_model(quantized_model, Path(config["work_dir"]) / ("quant_" + config["model_path"]))

    if shutil.which("dot"):
        utils.visualize(model, dataset, config["work_dir"])

    # evaluation phase
    utils.setup_logger(Path(config["work_dir"]) / "eval_metrics.log")

    X = []
    y = []
    for data_point in test_dataset:
        features, label = data_point
        X.append(features)  # Convert features to NumPy array
        y.append(label)

    utils.model_evaluation(model, X, y, config["dataset"], device, label="Original model",
                           sliding_window=config["sliding_window"])
    utils.model_evaluation(quantized_model, X, y, config["dataset"], device, label="Quantized model",
                           sliding_window=config["sliding_window"])

    if config["patgen"]:
        if config["patgen_force_rebuild"]:
            train_patgen(Path(config["work_dir"]) / "train_dataset.wlh", Path(config["work_dir"]) / "patgen",
                         "final_patterns.tex")
        eval_patgen(Path(config["work_dir"]) / "test_dataset.wlh",
                    Path(config["work_dir"]) / "patgen",
                    "patgen_mispredicted.txt",
                    "final_patterns.tex",
                    hyp_tf=dataset)


if __name__ == "__main__":
    main()
