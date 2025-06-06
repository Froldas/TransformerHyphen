from src.encoding import Encoding
from src.constants import VOWELS
from typing import Any, List


class SimpleFloatEncoding(Encoding):
    """
    Encodes all letters into float numbers. Space between encoding is based on the vocabulary size (unique letters)
    space = 1.0 / (vocab_size - 1)
    f.e. space = 0.1 for vocab size 11
    _ -> 0.0
    A -> 0.1 (0.0 + 1 * space)
    B -> 0.2 (0.0 + 2 * space)
    C -> 0.3 (0.0 + 3 * space)
    D -> 0.4 (0.0 + 4 * space)
    ...

    """

    def __init__(self, dataset: List[str], unique_letters: List[str]):
        self._letters = unique_letters
        self._letter_encoding = {}
        self._encoding_size = 1
        letter_count = len(self._letters)
        spacing_between_letters = 1.0 / (letter_count - 1)

        for idx, letter in enumerate(self._letters):
            # + 1 here is to distinguish between letters and empty space
            self.letter_encoding[letter] = [0.0 + (idx + 1) * spacing_between_letters]

    @property
    def letters(self) -> List[str]:
        return self._letters

    @property
    def letter_encoding(self) -> dict[str, List[Any]]:
        return self._letter_encoding

    @property
    def encoding_size(self) -> int:
        return self._encoding_size


class AdvancedFloatEncoding(Encoding):

    def __init__(self, dataset: List[str], unique_letters: List[str]):
        self._letters = unique_letters
        self._letter_encoding = {}
        self._encoding_size = 1

        letter_count = len(self._letters)

        spacing_ratio_between_vowels_consonants = 5
        spacing_between_letters = 1.0 / (letter_count - 1 + spacing_ratio_between_vowels_consonants)

        split_organized_letters = []
        vowel_count = 0
        for letter in self._letters:
            if letter in VOWELS:
                split_organized_letters.insert(0, letter)
                vowel_count += 1
            else:
                split_organized_letters.append(letter)

        for idx, letter in enumerate(split_organized_letters):
            if idx >= vowel_count:
                # add consonants (one extra space between consonants and vowels
                self.letter_encoding[letter] = [
                    0.0 + (idx + spacing_ratio_between_vowels_consonants) * spacing_between_letters]
            else:
                # add vowels
                self.letter_encoding[letter] = [0.0 - (idx + 1) * spacing_between_letters]

    @property
    def letters(self) -> List[str]:
        return self._letters

    @property
    def letter_encoding(self) -> dict[str, List[Any]]:
        return self._letter_encoding

    @property
    def encoding_size(self) -> int:
        return self._encoding_size


class AdvancedFloatEncoding2(Encoding):

    def __init__(self, dataset: List[str], unique_letters: List[str]):
        self._letters = unique_letters
        self._letter_encoding = {}
        self._encoding_size = 1

        letter_count = len(self._letters)

        spacing_ratio_between_vowels_consonants = 10
        spacing_between_letters = 1.0 / (letter_count - 1 + spacing_ratio_between_vowels_consonants)

        split_organized_letters = []
        vowel_count = 0
        for letter in self._letters:
            if letter in VOWELS:
                split_organized_letters.insert(0, letter)
                vowel_count += 1
            else:
                split_organized_letters.append(letter)

        for idx, letter in enumerate(split_organized_letters):
            if idx >= vowel_count:
                if letter == "l":
                    self.letter_encoding[letter] = [0.0 - 2 * spacing_between_letters]
                elif letter == "r":
                    self.letter_encoding[letter] = [0.0 - 3 * spacing_between_letters]
                else:
                    # add consonants (one extra space between consonants and vowels
                    self.letter_encoding[letter] = [1.0 - (idx - vowel_count) * spacing_between_letters]
            else:
                # add vowels
                self.letter_encoding[letter] = [-1.0 + idx * spacing_between_letters]

    @property
    def letters(self) -> List[str]:
        return self._letters

    @property
    def letter_encoding(self) -> dict[str, List[Any]]:
        return self._letter_encoding

    @property
    def encoding_size(self) -> int:
        return self._encoding_size
