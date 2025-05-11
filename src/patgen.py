import logging
import numpy as np
import os
import subprocess
import time
import tensorflow as tf
import torch

from pathlib import Path

from src.evaluation import report_metrics

# sequence of the
# * selector arguments triplets
# * patterns lengths pairs
# Length of the sequence defines amount of pattern levels

# format of each entry is: [(good, bad, threshold),(start, finish)]
#                            selector             , range
# More info about the entries: https://mirrors.nic.cz/tex-archive/info/patgen2-tutorial/patgen2-tutorial.pdf
PATGEN_SELECTORS = [[(1, 4, 20), (1, 3)],  #1 hyphenation
                     [(1, 3, 5), (1, 3)],  #2 inhibiting
                     [(1, 2, 2), (1, 4)],  #3 hyphenation
                     [(1, 2, 2), (1, 4)],  #4 inhibiting
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
    args += ["-p", str(Path(work_dir) / "patterns.tex")]
    args += ["-e", str(Path(work_dir) / "exceptions.tex")]
    args += [str(Path(work_dir) / output_filename)]
    subprocess.check_call(" ".join(args))


def eval_patgen(dataset, work_dir, output_filename, patterns_file, hyp_tf, measure_speed=False):
    tmp_dir = Path(work_dir)
    tmp_file = tmp_dir / "tmp_patterns"
    common_args = ["pypatgen", str(tmp_file)]

    full_out_file_pth = tmp_dir / output_filename
    Path.unlink(full_out_file_pth, missing_ok=True)

    # initialize dict for further processing
    eval_dict = {}
    with open(dataset, "r+", encoding="utf-8") as f:
        dataset_words = f.readlines()
        dataset_length = len(dataset_words)
        for dataset_word in dataset_words:
            #         input         = predicted
            eval_dict[dataset_word] = dataset_word

    # perform the evaluation
    args = common_args.copy()
    args += ["test"]
    args += [str(dataset)]
    args += ["-e", str(full_out_file_pth)]

    if measure_speed:
        start = time.time()
        output = subprocess.check_output(" ".join(args))
        end = time.time()
        logging.info(f"PyPatgen finished evaluation in{end - start: .2f} seconds")
        logging.info(f"PyPatgen has predicted{dataset_length / (end - start): .2f} words per second")
    else:
        output = subprocess.check_output(" ".join(args))

    logging.info(output.decode("utf-8").replace("\n", ""))

    with open(full_out_file_pth, "r+", encoding="utf-8") as f:
        mispredicted_words = f.readlines()

    mispredicted_labels = [word.replace("*", "").replace(".", "-") for word in mispredicted_words]
    mispredicted_predictions = [word.replace("*", "-").replace(".", "") for word in mispredicted_words]

    for idx in range(len(mispredicted_labels)):
        eval_dict[mispredicted_labels[idx]] = mispredicted_predictions[idx]

    ground_truth = []
    prediction = []

    for key, value in eval_dict.items():
        ground_truth += [hyp_tf.convert_word_to_expected_output(key)]

        hyphen_indices = [idx - 1 for idx, ch in enumerate(value) if ch == "-"]

        hyphen_expected = [0 for _ in range(hyp_tf.output_size)]
        for idx, hyphen_index in enumerate(hyphen_indices):
            hyphen_expected[hyphen_index - idx] = 1

        prediction += [tf.constant(hyphen_expected, dtype=tf.float32).numpy()]

    prediction = torch.tensor(np.array(prediction)).view(-1)
    ground_truth = torch.tensor(np.array(ground_truth)).view(-1)

    tp = (prediction == 1.0) & (ground_truth == 1.0)
    tn = (prediction == 0.0) & (ground_truth == 0.0)
    fp = (prediction == 1.0) & (ground_truth == 0.0)
    fn = (prediction == 0.0) & (ground_truth == 1.0)

    # Sum up
    stats = {
        "TP": tp.sum().item(),
        "TN": tn.sum().item(),
        "FP": fp.sum().item(),
        "FN": fn.sum().item()
    }
    patgen_size_kb = os.path.getsize(tmp_dir / patterns_file) / 1024
    report_metrics(stats, "Patgen", dataset, patgen_size_kb)
