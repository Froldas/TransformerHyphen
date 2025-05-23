from collections import Counter
import logging
import numpy as np
import os
import time
import torch

from pathlib import Path
from typing import List
from torch.utils.data import DataLoader


def model_size(model):
    torch.save(model.state_dict(), "temp.pth")
    size = os.path.getsize("temp.pth") / 1024  # Convert to KB
    os.remove("temp.pth")
    return size


def inference(model, dataloader, label, device, measure_time=False):
    start = time.time()

    x_pred = []

    for batch in dataloader:
        batch = batch.to(device)
        outputs = model.to(device)(batch)
        x_pred += outputs.cpu()

    if measure_time:
        end = time.time()
        logging.info(f"{label} finished evaluation in{end - start: .2f} seconds")
        logging.info(f"{label} has predicted{len(x_pred) / (end - start): .2f} words per second")

    return torch.Tensor(np.array(x_pred))

def model_evaluation(model, X, y, dataset, device, label="Full model", measure_speed=False):
    y = torch.Tensor(np.array(y)).detach().cpu()
    dataloader = DataLoader(X, batch_size=512, shuffle=False)
    x_pred = inference(model, dataloader, label, device, measure_time=measure_speed)

    tp = (x_pred.view(-1) == 1.0) & (y.view(-1) == 1.0)
    tn = (x_pred.view(-1) == 0.0) & (y.view(-1) == 0.0)
    fp = (x_pred.view(-1) == 1.0) & (y.view(-1) == 0.0)
    fn = (x_pred.view(-1) == 0.0) & (y.view(-1) == 1.0)

    # Sum up
    stats = {
        "TP": tp.sum().item(),
        "TN": tn.sum().item(),
        "FP": fp.sum().item(),
        "FN": fn.sum().item()
    }
    model_size_kb = model_size(model)
    report_metrics(stats, label, dataset, model_size_kb)


def convert_mispredicted(word, prediction, label):
    bad = (prediction == 1.0) & (label == 0.0)
    missed = (prediction == 0.0) & (label == 1.0)
    correct = (prediction == 1.0) & (label == 1.0)

    result = ""
    for i, char in enumerate(word.replace("-", "")):
        result += char
        if i < len(bad) and bad[i]:
            result += "*"
        if i < len(missed) and missed[i]:
            result += "."
        if i < len(missed) and correct[i]:
            result += "-"
    return result


def report_metrics(stats, model_label, dataset_path, model_size):
    missed = stats["FN"]
    bad = stats["FP"]
    correct = stats["TP"] # + stats["TN"]
    precision = stats["TP"] / (stats["TP"] + stats["FP"] + 0.00000001)
    recall = stats["TP"] / (stats["TP"] + stats["FN"] + 0.00000001)
    all = stats["TP"] + stats["TN"] + stats["FP"] + stats["FN"]
    accuracy = (stats["TP"] + stats["TN"]) / all

    # patgen convention? Assuming 100% = amount of hyphens in the original eval dataset
    total = correct + missed

    dataset_size_kb = os.path.getsize(dataset_path) / 1024

    logging.info(f"{model_label} evaluation: ")
    logging.info(f"    Dataset size is:{dataset_size_kb: .2f} KB")
    logging.info(f"    {model_label} size:{model_size: .2f} KB")
    logging.info(f"    {model_label} Effectivity:{(dataset_size_kb / model_size) * 100: .2f} %")

    logging.info(f"    {model_label} Accuracy:{accuracy: .4f}")
    logging.info(f"    {model_label} Recall:{recall: .4f}")
    logging.info(f"    {model_label} Precision:{precision: .4f}")

    logging.info(f"    {model_label} Correct Hyphens: {correct} ({(correct * 100 / total):.2f} %)")
    logging.info(f"    {model_label} Bad Hyphens: {bad} ({(bad * 100 / total):.2f} %)")
    logging.info(f"    {model_label} Missed Hyphens: {missed} ({(missed * 100 / total):.2f} %)")


def analyze_mismatches(config):
    model_misprediction_pth = Path(config["work_dir"]) / config["mispredict_path"]
    patgen_misprediction_pth = Path(config["work_dir"]) / "patgen" / "patgen_mispredicted.txt"

    with open(model_misprediction_pth, "r+", encoding="utf-8") as f:
        model_mispredicted = f.readlines()

    with open(patgen_misprediction_pth, "r+", encoding="utf-8") as f:
        patgen_mispredicted = f.readlines()

    mismatches_dict = {}

    for mismatch in (model_mispredicted + patgen_mispredicted):
        mismatch_key = mismatch.replace("*", "").replace(".", "").replace("\n", "")
        if mismatch_key in mismatches_dict:
            mismatches_dict[mismatch_key].append(mismatch.replace("\n", ""))
        else:
            mismatches_dict[mismatch_key] = [mismatch.replace("\n", "")]

    both_misprediction_pth = Path(config["work_dir"]) / "common_mispredicted.txt"

    commonly_mispredicted = []
    with open(both_misprediction_pth, "w+", encoding="utf-8") as f:
        f.writelines(f"* = bad hyphen\n"
                     f". = missing hyphen\n"
                     f"model|patgen\n")
        for key, value in mismatches_dict.items():
            if len(value) == 2:
                line = f"{value[0]}|{value[1]}"
                f.writelines(f"{line}\n")
                commonly_mispredicted.append(line)

    analyze_substrings(commonly_mispredicted)



def analyze_substrings(words: List[str], min_sub_len: int = 3, max_sub_len: int = 5, top_n: int = 10) -> None:
    """
    Analyzes a list of words to find the most common substrings containing '*' or '.'.

    Parameters:
    - words: List of input words.
    - min_sub_len: Minimum length of substrings to consider.
    - max_sub_len: Maximum length of substrings to consider.
    - top_n: Number of top patterns to display.
    """
    substr_counter = Counter()

    for word in words:
        word_len = len(word)
        lower_word = word.lower()

        # Substrings containing '*' or '.'
        for length in range(min_sub_len, max_sub_len + 1):
            for i in range(word_len - length + 1):
                substr = lower_word[i:i+length]
                if '*' in substr or '.' in substr:
                    substr_counter[substr] += 1

    # Display Results
    logging.info(f"Common mismatch analysis:\n  top {top_n} Substrings Containing '*' or '.':")
    for substr, count in substr_counter.most_common(top_n):
        logging.info(f"    {substr}: {count}")
