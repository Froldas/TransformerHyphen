import logging
import numpy as np
import os
import shutil
import sys
import torch
from pathlib import Path

import torch.nn as nn
import torch.optim as optim

import src.utils as utils
from src.ConfDict import Models, Encodings
from src.evaluation import model_evaluation, convert_mispredicted, analyze_mismatches
from src.dataset import HyphenationDataset, HyphenationDatasetSlidingWindow
from src.patgen import train_patgen, eval_patgen
from src.training import model_training

YML_CONF_PATH = "configuration.yml"


def evaluation(model, quantized_model, config, test_dataset, original_dataset, device):
    # evaluation phase
    utils.setup_logger(Path(config["work_dir"]) / "eval_metrics.log")

    X = []
    y = []
    original_words = []
    for index in test_dataset.indices:
        features, label = original_dataset[index]
        X.append(features)  # Convert features to NumPy array
        y.append(label)
        if config["generate_mispredicted"]:
            original_words.append(original_dataset.words[index])

    model_evaluation(model, X, y, config["dataset"], device, label="Original model",
                     measure_speed=config["measure_speed"])
    model_evaluation(quantized_model, X, y, config["dataset"], "cpu", label="Quantized model",
                     measure_speed=config["measure_speed"])

    if config["generate_mispredicted"]:
        with open(Path(config["work_dir"]) / config["mispredict_path"], "w+", encoding="utf-8") as f:
            x_pred = model(torch.Tensor(np.array(X)).to("cpu"))
            for i in range(len(original_words)):
                if not torch.equal(x_pred[i], torch.Tensor(y[i])):
                    f.writelines(
                        f"{convert_mispredicted(original_words[i], x_pred[i], torch.Tensor(y[i]))}\n")


def quantize_and_save(model, config, dataset):
    # Dump trained model
    quantized_model = utils.quantize_model(model.to("cpu"))
    utils.save_model(model, Path(config["work_dir"]) / config["model_path"])
    utils.save_model(quantized_model, Path(config["work_dir"]) / ("quant_" + config["model_path"]))

    if shutil.which("dot"):
        utils.visualize_model(model, dataset, config["work_dir"])
    return quantized_model


def run_patgen(config, dataset):
    patgen_path = Path(config["work_dir"]) / "patgen"
    if not Path.is_dir(patgen_path) or config["patgen_force_rebuild"]:
        train_patgen(Path(config["work_dir"]) / "train_dataset.wlh", patgen_path, "final_patterns.tex")
    eval_patgen(Path(config["work_dir"]) / "test_dataset.wlh", patgen_path,
                "patgen_mispredicted.txt",
                "patterns.tex",
                hyp_tf=dataset,
                measure_speed=config["measure_speed"])


def merge_english_words(config):
    eng_words = utils.generate_hyphenated_english_words(Path("datasets") / 'Oxford 5000.txt')
    merged_dataset_path = Path(config["work_dir"]) / "merged_dataset.wlh"
    utils.append_dataset(Path(config["dataset"]), eng_words, merged_dataset_path)
    return merged_dataset_path


def setup_and_set_device(config):
    utils.setup_logger(Path(config["work_dir"]) / config["training_log_path"])
    # set seed for reproducibility
    utils.set_seed(config["seed"])
    # Qunatized models cannot use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    return device

def export_onxx_model(model, model_input,  config):
    torch.onnx.export(model,  # model being run
                      model_input,  # model input (or a tuple for multiple inputs)
                      Path(config["work_dir"]) / config["onxx_model_path"],  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})

def main():
    if len(sys.argv) > 1:
        # use config given as a parameter (needed by grid_run)
        config = utils.load_yaml_conf(Path(sys.argv[1]))
    else:
        config = utils.load_yaml_conf(Path(YML_CONF_PATH))

    os.makedirs(config["work_dir"], exist_ok=True)

    device = setup_and_set_device(config)

    if config["english_words"]:
        config["dataset"] = merge_english_words(config)

    # Create datasets and dataloaders
    if config["sliding_window"]:
        dataset_type = HyphenationDatasetSlidingWindow
    else:
        dataset_type = HyphenationDataset

    dataset = dataset_type(data_file=config["dataset"],
                           work_dir=config["work_dir"],
                           encoding=Encodings().encodings[config["encoding"]],
                           print_info=config["print_dataset_statistics"])

    train_dataset, test_dataset = utils.split_dataset(dataset, config["train_split"],
                                                      work_dir=config["work_dir"],
                                                      dump_datasets=config["patgen"],
                                                      seed=config["seed"])

    model = Models(dataset.num_input_tokens,
                   dataset.encoding_size,
                   dataset.output_size,
                   config["hyphen_threshold"]).models[config["model"]].to(device)

    loss_func = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.05)

    # Training
    model_training(model,
                   train_dataset,
                   config["num_epochs"],
                   config["num_folds"],
                   optimizer,
                   loss_func,
                   config["batch_size"],
                   device,
                   config["work_dir"],
                   config["seed"],
                   config["early_stopping"])

    # quantize and dump both original and quantized version
    quantized_model = quantize_and_save(model, config, dataset)
    model.to(device)
    quantized_model.to(device)

    evaluation(model, quantized_model, config, test_dataset, dataset, device)

    if config["patgen"]:
        run_patgen(config, dataset)

    if config["analyze_mismatches"]:
        analyze_mismatches(config)

    if config["onxx_export"]:
        model_input = torch.randn(dataset.input_size * dataset.encoding_size, requires_grad=True)
        export_onxx_model(model, model_input, config)

if __name__ == "__main__":
    main()
