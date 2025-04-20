import math
import numpy as np

from src.encoding import Encoding
from src.constants import VOWELS
from typing import Any, List


class BinaryEncoding(Encoding):
    """
    Encodes all letters into n-bits binary numbers. Size of the numbers depends on the vocabulary size (unique letters)
    _ -> 0b00000
    A -> 0b00001
    B -> 0b00010
    C -> 0b00011
    D -> 0b00100
    ...

    """

    def __init__(self, dataset: List[str], unique_letters: List[str]):
        self._letters = unique_letters
        self._letter_encoding = {}
        self._encoding_size = math.ceil(math.log2(len(self._letters) + 1))
        for idx, letter in enumerate(self._letters):
            # + 1 here is to distinguish between letters and empty space
            self.letter_encoding[letter] = [int(bit) for bit in np.binary_repr(idx + 1, width=self.encoding_size)]

    @property
    def letters(self) -> List[str]:
        return self._letters

    @property
    def letter_encoding(self) -> dict[str, List[Any]]:
        return self._letter_encoding

    @property
    def encoding_size(self) -> int:
        return self._encoding_size

class AdvancedBinaryEncoding(Encoding):
    """
    Encodes all letters into n-bits binary numbers. Size of the numbers depends on the vocabulary size (unique letters)
    Vowels and Consonants are split in the space so that there are further away from each other
    _ -> 0b00000
    A -> 0b00001
    B -> 0b00010
    C -> 0b00011
    D -> 0b00100
    ...

    """

    def __init__(self, dataset: List[str], unique_letters: List[str]):
        self._letters = unique_letters
        self._letter_encoding = {}
        self._encoding_size = math.ceil(math.log2(len(self._letters) + 1))

        split_organized_letters = []
        vowel_count = 0
        for letter in self._letters:
            if letter in VOWELS:
                split_organized_letters.insert(0, letter)
                vowel_count += 1
            elif letter in ["l", "r"]:
                split_organized_letters.insert(vowel_count, letter)
            else:
                split_organized_letters.append(letter)

        for idx, letter in enumerate(split_organized_letters):
            # + 1 here is to distinguish between letters and empty space
            self.letter_encoding[letter] = [int(bit) for bit in np.binary_repr(idx + 1, width=self.encoding_size)]

    @property
    def letters(self) -> List[str]:
        return self._letters

    @property
    def letter_encoding(self) -> dict[str, List[Any]]:
        return self._letter_encoding

    @property
    def encoding_size(self) -> int:
        return self._encoding_size

class OneHotEncoding(Encoding):
    """
    Encodes all letters into one-hot vector. Size of the numbers depends on the vocabulary size (unique letters)
    _ -> ...00000
    A -> ...00001
    B -> ...00010
    C -> ...00100
    D -> ...01000
    ...

    """
    def __init__(self, dataset: List[str], unique_letters: List[str]):
        self._letters = unique_letters
        self._letter_encoding = {}
        self._encoding_size = len(self._letters)
        for idx, letter in enumerate(self._letters):
            self.letter_encoding[letter] = [1 if i == idx else 0 for i in range(self._encoding_size)]

    @property
    def letters(self) -> List[str]:
        return self._letters

    @property
    def letter_encoding(self) -> dict[str, List[Any]]:
        return self._letter_encoding

    @property
    def encoding_size(self) -> int:
        return self._encoding_size
