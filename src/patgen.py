import os
import subprocess
from pathlib import Path

# sequence of the
# * selector arguments triplets
# * patterns lengths pairs
# Length of the sequence defines amount of pattern levels

# format of each entry is: [(good, bad, threshold),(start, finish)]
# More info about the entries: https://mirrors.nic.cz/tex-archive/info/patgen2-tutorial/patgen2-tutorial.pdf
PATTENN_SELECTORS = [[(1, 4, 20), (1, 3)],  #1 hyphenation
                     [(1, 2, 5), (1, 3)],  #2 inhibiting
                     [(1, 2, 5), (1, 4)],  #3 hyphenation
                     [(1, 1, 2), (1, 4)],  #4 inhibiting
                     [(1, 1, 2), (1, 5)],  #5 hyphenation
                     [(1, 1, 1), (1, 5)],  #3 inhibiting
                     [(1, 1, 1), (1, 6)],  #7 hyphenation
                     [(1, 1, 1), (1, 6)],  #8 inhibiting
                     ]


def train_patgen(dataset, work_dir, output_filename):

    tmp_dir = Path(work_dir) / "patgen"
    tmp_file = tmp_dir / "tmp_patterns"

    os.makedirs(tmp_dir, exist_ok=True)
    Path.unlink(tmp_file, missing_ok=True)

    common_args = ["pypatgen", str(tmp_file)]

    args = common_args.copy()
    args += ["new"]
    args += [dataset]
    subprocess.check_call(" ".join(args))

    for selector, lengths in PATTENN_SELECTORS:
        args = common_args.copy()
        args += ["train"]
        args += ["-r", f"{lengths[0]}-{lengths[1]}"]
        args += ["-s", f"{selector[0]}:{selector[1]}:{selector[2]}"]
        args += ["-c"]
        subprocess.check_call(" ".join(args))

    args = common_args.copy()
    args += ["export"]
    args += [str(Path(work_dir) / output_filename)]
    subprocess.check_call(" ".join(args))


def evaluate_patterns(patterns_path, evaluation_dataset):
    pass
