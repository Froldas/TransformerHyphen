from src.encoding import Encoding
from typing import Any


class SimpleEmbedding(Encoding):
    """
    Encodes all letters into embeddings
    space = 1.0 / (vocab_size - 1)
    f.e. space = 0.1 for vocab size 11
    _ -> 0.0
    A -> 0.1 (0.0 + 1 * space)
    B -> 0.2 (0.0 + 2 * space)
    C -> 0.3 (0.0 + 3 * space)
    D -> 0.4 (0.0 + 4 * space)
    ...

    """
    def __init__(self, dataset: [str], unique_letters: [str]):
        self._letters = unique_letters
        self._letter_encoding = {}
        self._encoding_size = 1
        letter_count = len(self._letters)
        spacing_between_letters = 1.0 / (letter_count - 1)

        for idx, letter in enumerate(self._letters):
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
