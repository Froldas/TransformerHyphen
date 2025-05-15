import logging
import os
import pickle
import numpy as np
import tensorflow as tf

from pathlib import Path
from torch.utils.data import Dataset

from src.utils import remove_hyphenation, sliding_splits
from src.constants import HYPHENS


class HyphenationInterface:
    def __init__(self, num_input_tokens, encoding_size, output_size, letter_encoding, work_dir, pos_embedding=False):
        self.num_input_tokens = num_input_tokens
        self.encoding_size = encoding_size
        self.input_size = num_input_tokens * encoding_size
        self.output_size = output_size
        self.letter_encoding = letter_encoding
        self.work_dir = work_dir
        self.pos_embedding = pos_embedding

    def encode(self, word):
        input_vector = [0 for _ in range(self.input_size)]

        for idx, letter in enumerate(word):
            if letter not in list(self.letter_encoding.keys()):
                continue
            input_vector[idx * self.encoding_size: (idx + 1) * self.encoding_size] = self.letter_encoding[letter]
        encoded_input_tensor = tf.constant(input_vector, dtype=tf.float32).numpy()
        if self.pos_embedding:
            encoded_input_tensor = self.add_positional_encoding(encoded_input_tensor)
        return encoded_input_tensor

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
                "encoding_size": self.encoding_size,
                "output_size": self.output_size,
                "letter_encoding": self.letter_encoding}

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

    def add_positional_encoding(self, x):
        """
        Adds sinusoidal positional encoding to the input NumPy array.

        Args:
            x (np.ndarray): Input array of shape (batch_size, seq_len, embedding_dim)

        Returns:
            np.ndarray: Array with positional encoding added, same shape as input.
        """
        # Create a matrix of positions (seq_len, 1)
        position = np.arange(self.num_input_tokens).reshape(self.num_input_tokens, 1)
        div_term = np.exp(np.arange(0, self.encoding_size, 2) * (-np.log(10000.0) / self.encoding_size))

        pe = np.zeros((self.num_input_tokens, self.encoding_size), dtype=np.float32)

        pe[:, 0::2] = np.sin(position * div_term)
        if self.encoding_size % 2 == 0:
            pe[:, 1::2] = np.cos(position * div_term)
        else:
            pe[:, 1::2] = np.cos(position * div_term[:(self.encoding_size // 2)])

        # Expand pe to match the batch size and add to input
        x = x + pe.flatten(order="C")  # Broadcasting addition

        return x


class HyphenationDataset(Dataset, HyphenationInterface):
    def __init__(self, data_file, work_dir, encoding=None, print_info=False, pos_embedding=False):
        self.data_file = data_file
        self.unique_letters = set()
        self.longest_word = ""
        self.words = []
        self.letter_encoding = {}
        self.datapoints = []

        self._read_dataset(data_file)

        self.encoding = encoding(self.words, self.unique_letters)
        self.letter_encoding = self.encoding.letter_encoding
        self.encoding_size = self.encoding.encoding_size

        self.input_size = len(self.longest_word) * self.encoding_size
        self.output_size = len(self.longest_word) - 1

        super().__init__(
            len(self.longest_word),
            self.encoding_size,
            self.output_size,
            self.letter_encoding,
            work_dir,
            pos_embedding=pos_embedding)

        for word in self.words:
            word_without_hyphens = word.replace("-", "")
            input_vector = self.encode(word_without_hyphens)
            label = self.convert_word_to_expected_output(word)
            self.datapoints.append((input_vector, label))

        if print_info:
            self._print_info()

        self._dump_configuration()

    def _read_dataset(self, data_file_path):
        with open(data_file_path, 'r', encoding='utf-8') as f:
            for word in f:
                # remove trailing space and convert to lowercase
                word = word.strip().lower()

                for hyphen in HYPHENS:
                    word = word.replace(hyphen, "-")

                self.words.append(word)
                word_without_hyphens = remove_hyphenation(word)

                self.unique_letters.update(list(word_without_hyphens))
                self.longest_word = max(self.longest_word, word_without_hyphens, key=len)
        self.unique_letters = sorted(list(self.unique_letters))

    def _print_info(self):
        logging.info(f"Dataset size: {self.__len__()}")
        logging.info(f"Longest word: {self.longest_word}")
        logging.info(f"Longest word size: {len(self.longest_word)}")
        logging.info(f"Input_size: {self.input_size}")
        logging.info(f"Unique letters: {sorted(self.unique_letters)}")
        logging.info(f"Number of unique letters: {len(self.unique_letters)}")
        logging.info(f"Letter encoding: {self.letter_encoding}")

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        return self.datapoints[idx]


class HyphenationDatasetSlidingWindow(Dataset, HyphenationInterface):
    def __init__(self, data_file, work_dir, encoding=None, context_size=4, print_info=False, pos_embedding=False):
        self.data_file = data_file
        self.unique_letters = set()
        self.longest_word = ""
        self.words = []
        self.letter_encoding = {}
        self.datapoints = []

        window_size = 2 * context_size + 1

        self._read_dataset(data_file)
        self.encoding = encoding(self.words, self.unique_letters)
        self.letter_encoding = self.encoding.letter_encoding
        self.encoding_size = self.encoding.encoding_size

        self.input_size = window_size * self.encoding_size
        self.output_size = 1

        super().__init__(
            window_size,
            self.encoding_size,
            self.output_size,
            self.letter_encoding,
            work_dir,
            pos_embedding=pos_embedding)

        for word in self.words:
            chunks = sliding_splits(word)
            for input_substring, label in chunks:
                input_vector = self.encode(remove_hyphenation(input_substring))
                self.datapoints.append((input_vector, label))

        if print_info:
            self._print_info()

        self._dump_configuration()

    def _read_dataset(self, data_file_path):
        with open(data_file_path, 'r', encoding='utf-8') as f:
            for word in f:
                # remove trailing space and convert to lowercase
                word = word.strip().lower()

                for hyphen in HYPHENS:
                    word = word.replace(hyphen, "-")

                self.words.append(word)
                word_without_hyphens = remove_hyphenation(word)

                self.unique_letters.update(list(word_without_hyphens))
                self.longest_word = max(self.longest_word, word_without_hyphens, key=len)

    def _print_info(self):
        logging.info(f"Dataset size: {len(self.datapoints)}")
        logging.info(f"Longest word: {self.longest_word}")
        logging.info(f"Longest word size: {len(self.longest_word)}")
        logging.info(f"Input_size: {self.input_size}")
        logging.info(f"Unique letters: {sorted(self.unique_letters)}")
        logging.info(f"Number of unique letters: {len(self.unique_letters)}")
        logging.info(f"Letter encoding: {self.letter_encoding}")

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        return self.datapoints[idx]
