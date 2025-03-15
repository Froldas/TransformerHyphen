import math
import numpy as np

from src.encoding import Encoding
from typing import Any


class BinaryEncoding(Encoding):

    def __init__(self, letters: [str]):
        self._letters = letters
        self._letter_encoding = {}
        self._encoding_size = math.ceil(math.log2(len(letters) + 1))
        for idx, letter in enumerate(letters):
            # + 1 here is to distinguish between letters and empty space
            self.letter_encoding[letter] = [int(bit) for bit in np.binary_repr(idx + 1, width=self.encoding_size)]

    @property
    def letters(self) -> [str]:
        return self._letters

    @property
    def letter_encoding(self) -> dict[str, Any]:
        return self._letter_encoding

    @property
    def encoding_size(self) -> int:
        return self._encoding_size
