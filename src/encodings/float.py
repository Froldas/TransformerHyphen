import math
import numpy as np

from src.encoding import Encoding
from typing import Any


class FloatEncoding(Encoding):

    def __init__(self, letters: [str]):
        self._letters = letters
        self._letter_encoding = {}
        self._encoding_size = 1
        letter_count = len(letters)
        spacing_between_letters = 1.0 / (letter_count - 1)

        for idx, letter in enumerate(letters):
            # + 1 here is to distinguish between letters and empty space
            self.letter_encoding[letter] = [0.0 + (idx + 1) * spacing_between_letters]

    @property
    def letters(self) -> [str]:
        return self._letters

    @property
    def letter_encoding(self) -> dict[str, [Any]]:
        return self._letter_encoding

    @property
    def encoding_size(self) -> int:
        return self._encoding_size
