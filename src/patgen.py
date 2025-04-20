import os
import logging
import subprocess
import torch
import numpy as np
from pathlib import Path
import tensorflow as tf
from src.utils import remove_hyphenation
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# sequence of the
# * selector arguments triplets
# * patterns lengths pairs
# Length of the sequence defines amount of pattern levels

# format of each entry is: [(good, bad, threshold),(start, finish)]
# More info about the entries: https://mirrors.nic.cz/tex-archive/info/patgen2-tutorial/patgen2-tutorial.pdf
PATGEN_SELECTORS = [[(1, 4, 20), (1, 3)],  #1 hyphenation
                     [(1, 2, 5), (1, 3)],  #2 inhibiting
                     [(1, 2, 5), (1, 4)],  #3 hyphenation
                     [(1, 1, 2), (1, 4)],  #4 inhibiting
                     #[(1, 1, 2), (1, 5)],  #5 hyphenation
                     #[(1, 1, 1), (1, 5)],  #3 inhibiting
                    # [(1, 1, 1), (1, 6)],  #7 hyphenation
                    # [(1, 1, 1), (1, 6)],  #8 inhibiting
                     ]


def train_patgen(dataset, work_dir, output_filename):
    tmp_dir = Path(work_dir)
    tmp_file = tmp_dir / "tmp_patterns"
    common_args = ["pypatgen", str(tmp_file)]

    os.makedirs(tmp_dir, exist_ok=True)
    Path.unlink(tmp_file, missing_ok=True)

    args = common_args.copy()
    args += ["new"]
    args += [str(dataset)]
    subprocess.check_call(" ".join(args))

    for selector, lengths in PATGEN_SELECTORS:
        args = common_args.copy()
        args += ["train"]
        args += ["-r", f"{lengths[0]}-{lengths[1]}"]
        args += ["-s", f"{selector[0]}:{selector[1]}:{selector[2]}"]
        args += ["-c"]
        subprocess.check_call(" ".join(args))

    Path.unlink(Path(work_dir) / output_filename, missing_ok=True)

    args = common_args.copy()
    args += ["export"]
    args += [str(Path(work_dir) / output_filename)]
    subprocess.check_call(" ".join(args))


def eval_patgen(dataset, work_dir, output_filename, patterns_file, hyp_tf):
    tmp_dir = Path(work_dir)
    tmp_file = tmp_dir / "tmp_patterns"
    common_args = ["pypatgen", str(tmp_file)]

    full_out_file_pth = tmp_dir / output_filename
    Path.unlink(full_out_file_pth, missing_ok=True)

    args = common_args.copy()
    args += ["test"]
    args += [str(dataset)]
    args += ["-e", str(full_out_file_pth)]
    output = subprocess.check_output(" ".join(args))
    logging.info(output.decode("utf-8"))
    eval_dict = {}

    with open(full_out_file_pth, "r+", encoding="utf-8") as f:
        mispredicted_words = f.readlines()

    with open(dataset, "r+", encoding="utf-8") as f:
        dataset_words = f.readlines()
        for dataset_word in dataset_words:
            # key=input,  = predicted
            eval_dict[dataset_word] = dataset_word

    mispredicted_unified = [word.replace("*", "=").replace(".", "=") for word in mispredicted_words]

    for idx in range(len(mispredicted_unified)):
        eval_dict[mispredicted_unified[idx]] = mispredicted_words[idx]

    ground_truth = []
    prediction = []

    for key, value in eval_dict.items():
        ground_truth += [hyp_tf.convert_word_to_expected_output(key)]

        hyphen_indices = [idx - 1 for idx, ch in enumerate(value) if ch == "-" or ch == "*"]

        hyphen_expected = [0 for _ in range(hyp_tf.output_size)]
        for idx, hyphen_index in enumerate(hyphen_indices):
            hyphen_expected[hyphen_index - idx] = 1

        prediction += [tf.constant(hyphen_expected, dtype=tf.float32).numpy()]

    accuracy = accuracy_score(ground_truth, prediction)
    recall = recall_score(ground_truth, prediction, average="samples", zero_division=0.0)
    precision = precision_score(ground_truth, prediction, average="samples", zero_division=0.0)
    f1 = f1_score(ground_truth, prediction, average="samples", zero_division=0.0)

    dataset_size_kb = os.path.getsize(hyp_tf.data_file) / 1024
    patgen_size_kb = os.path.getsize(tmp_dir / patterns_file) / 1024
    logging.info(f"Patgen evaluation: ")
    logging.info(f"    Dataset size is: {dataset_size_kb:.2f} KB")
    logging.info(f"    Patgen pattern size: {patgen_size_kb:.2f} KB")
    logging.info(f"    Patgen Efficiency: {(dataset_size_kb / patgen_size_kb) * 100:.2f} %")
    logging.info(f"    Patgen Accuracy: {accuracy:.4f}")
    logging.info(f"    Patgen Recall: {recall:.4f}")
    logging.info(f"    Patgen Precision: {precision:.4f}")
    logging.info(f"    Patgen F1: {f1:.4f}")







