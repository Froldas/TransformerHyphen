import math
import os
from pathlib import Path
import numpy as np
import pickle

import tensorflow as tf
from torch.utils.data import Dataset

class HyphenationInterace():
    def __init__(self, longest_word, bits_per_letter, input_size, output_size, letter_encoding, work_dir):
        self.longest_word = longest_word
        self.bits_per_letter = bits_per_letter
        self.input_size = input_size
        self.output_size = output_size
        self.letter_encoding = letter_encoding
        self.work_dir = work_dir

    def convert_word_to_input_tensor(self, word):
        input_vector = [0 for _ in range(self.input_size)]

        for idx, letter in enumerate(word):
            if letter not in list(self.letter_encoding.keys()):
                continue
            input_vector[idx * self.bits_per_letter: (idx+1) * self.bits_per_letter] = self.letter_encoding[letter]
        return tf.constant([input_vector], dtype=tf.float32).numpy()

    def convert_word_to_expected_output(self, word):
        hyphen_indices = [idx - 1 for idx, ch in enumerate(word) if ch == "-"]

        hyphen_expected = [0 for _ in range(self.output_size)]
        for idx, hyphen_index in enumerate(hyphen_indices):
            hyphen_expected[hyphen_index - idx] = 1

        expected_output = tf.constant(hyphen_expected, dtype=tf.float32).numpy()

        return expected_output
    @property
    def conf_path(self):
        return Path(self.work_dir) / "conf.pk"

    def _dump_configuration(self):
        data = {}
        data["longest_word"] = self.longest_word
        data["bits_per_letter"] = self.bits_per_letter
        data["input_size"] = self.input_size
        data["output_size"] = self.output_size
        data["letter_encoding"] = self.letter_encoding

        os.makedirs(Path(self.work_dir), exist_ok=True)

        with open(self.conf_path, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def load_configuration(work_dir, conf_path):
        with open(Path("build") / "conf.pk", "rb") as f:
            data = pickle.load(f)
        return HyphenationInterace(data["longest_word"],
                                   data["bits_per_letter"],
                                   data["input_size"],
                                   data["output_size"],
                                   data["letter_encoding"],
                                   Path("build"))

class HyphenationDataset(Dataset, HyphenationInterace):
    def __init__(self, data_file, work_dir, print_info=False):
        self.unique_letters = set()
        self.longest_word = ""
        self.words = []
        self.letter_encoding = {}
        self.datapoints = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                self.words.append(word)
                word_without_hyphens = word.replace("-", "")

                self.unique_letters.update(list(word_without_hyphens))
                self.longest_word = max(self.longest_word, word_without_hyphens, key=len)

        self.num_unique_letters = len(self.unique_letters)
        # need one extra space for empty place
        self.bits_per_letter = math.ceil(math.log2(self.num_unique_letters + 1))
        self.input_size = len(self.longest_word) * self.bits_per_letter
        self.output_size = len(self.longest_word) - 1

        for idx, letter in enumerate(sorted(self.unique_letters)):
            # + 1 here is to distinguish between letters and empty space
            self.letter_encoding[letter] = [int(bit) for bit in np.binary_repr(idx + 1, width=self.bits_per_letter)]
        if print_info:
            print(f"Input_size: {self.input_size}")
            print(f"Number of unique letters: {self.num_unique_letters}")
            print(f"Unique letters: {sorted(self.unique_letters)}")
            print(f"Longest word: {self.longest_word}")
            print(f"Letter encoding: {self.letter_encoding}")
        super().__init__(
            self.longest_word,
            self.bits_per_letter,
            self.input_size,
            self.output_size,
            self.letter_encoding,
            work_dir)

        for word in self.words:
            word_without_hyphens = word.replace("-", "")
            input_vector = self.convert_word_to_input_tensor(word_without_hyphens)
            label = self.convert_word_to_expected_output(word)
            self.datapoints.append((input_vector, label))

        self._dump_configuration()

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        return self.datapoints[idx]

def insert_hyphenation(string, bit_list):
    # Convert the string to a list for easier manipulation
    string_list = list(string)

    # Iterate over the bit_list and insert '-' where there is a 1
    hyphens_inserted = 0
    for index, bit in enumerate(bit_list):
        if bit == 1:
            # Insert '-' at the predicted spot
            string_list.insert(index + hyphens_inserted + 1, '-')
            hyphens_inserted += 1

    # Convert the list back to a string
    result_string = ''.join(string_list)
    return result_string
