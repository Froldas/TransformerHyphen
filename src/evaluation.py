
import logging
import numpy as np
import os

import torch

def model_size(model):
    torch.save(model.state_dict(), "temp.pth")
    size = os.path.getsize("temp.pth") / 1024  # Convert to KB
    os.remove("temp.pth")
    return size


def model_evaluation(model, X, y, dataset, device, label="Full model", sliding_window=False):
    y = torch.Tensor(np.array(y)).detach()
    x_pred = model.cpu()((torch.Tensor(np.array(X))).cpu())

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

    missed = stats["FN"]
    bad = stats["FP"]
    correct = stats["TP"] + stats["TN"]
    precision = stats["TP"] / (stats["TP"] + stats["FP"])
    recall = stats["TP"] / (stats["TP"] + stats["FN"])
    total = stats["TP"] + stats["TN"] + stats["FP"] + stats["FN"]
    accuracy = (stats["TP"] + stats["TN"]) / total

    dataset_size_kb = os.path.getsize(dataset) / 1024
    model_size_kb = model_size(model)
    logging.info(f"{label} evaluation: ")
    logging.info(f"    Dataset size is:{dataset_size_kb: .2f} KB")
    logging.info(f"    {label} size:{model_size_kb: .2f} KB")
    logging.info(f"    {label} Efficiency:{(dataset_size_kb / model_size_kb) * 100: .2f}%")

    logging.info(f"    {label} Accuracy:{accuracy: .4f}")
    logging.info(f"    {label} Recall:{recall: .4f}")
    logging.info(f"    {label} Precision:{precision: .4f}")

    logging.info(f"    {label} Correct Hyphens: {correct} ({(correct * 100 / total):.2f}%)")
    logging.info(f"    {label} Bad Hyphens: {bad} ({(bad * 100 / total):.2f}%)")
    logging.info(f"    {label} Missed Hyphens: {missed} ({(missed * 100 / total):.2f}%)")

