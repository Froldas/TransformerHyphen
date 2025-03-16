import logging
import os
import pickle

from pathlib import Path
import tensorflow as tf
from torch.utils.data import Dataset


class HyphenationInterface:
    def __init__(self, num_input_tokens, encoding_size, output_size, letter_encoding, work_dir):
        self.num_input_tokens = num_input_tokens
        self.encoding_size = encoding_size
        self.input_size = num_input_tokens * encoding_size
        self.output_size = output_size
        self.letter_encoding = letter_encoding
        self.work_dir = work_dir

    def encode(self, word):
        input_vector = [0 for _ in range(self.input_size)]

        for idx, letter in enumerate(word):
            if letter not in list(self.letter_encoding.keys()):
                continue
            input_vector[idx * self.encoding_size: (idx + 1) * self.encoding_size] = self.letter_encoding[letter]
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
        data = {"num_input_tokens": self.num_input_tokens,
                "encoding_size"   : self.encoding_size,
                "output_size"     : self.output_size,
                "letter_encoding" : self.letter_encoding}

        os.makedirs(Path(self.work_dir), exist_ok=True)

        with open(self.conf_path, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def load_configuration(work_dir, conf_path):
        with open(Path(work_dir) / conf_path, "rb") as f:
            data = pickle.load(f)
        return HyphenationInterface(data["num_input_tokens"],
                                   data["encoding_size"],
                                   data["output_size"],
                                   data["letter_encoding"],
                                   Path("build"))


class HyphenationDataset(Dataset, HyphenationInterface):
    def __init__(self, data_file, work_dir, encoding=None, print_info=False):
        self.unique_letters = set()
        self.longest_word = ""
        self.words = []
        self.letter_encoding = {}
        self.datapoints = []

        self._read_dataset(data_file)

        self.encoding = encoding(sorted(self.unique_letters))
        self.letter_encoding = self.encoding.letter_encoding
        self.encoding_size = self.encoding.encoding_size

        self.input_size = len(self.longest_word) * self.encoding_size
        self.output_size = len(self.longest_word) - 1

        if print_info:
            self._print_info()

        super().__init__(
            len(self.longest_word),
            self.encoding_size,
            self.output_size,
            self.letter_encoding,
            work_dir)

        for word in self.words:
            word_without_hyphens = word.replace("-", "")
            input_vector = self.encode(word_without_hyphens)
            label = self.convert_word_to_expected_output(word)
            self.datapoints.append((input_vector, label))

        self._dump_configuration()

    def _read_dataset(self, data_file_path):
        with open(data_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                self.words.append(word)
                word_without_hyphens = word.replace("-", "")

                self.unique_letters.update(list(word_without_hyphens))
                self.longest_word = max(self.longest_word, word_without_hyphens, key=len)

    def _print_info(self):
        logging.info(f"Input_size: {self.input_size}")
        logging.info(f"Number of unique letters: {len(self.unique_letters)}")
        logging.info(f"Unique letters: {sorted(self.unique_letters)}")
        logging.info(f"Longest word: {self.longest_word}")
        logging.info(f"Letter encoding: {self.letter_encoding}")

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        return self.datapoints[idx]
