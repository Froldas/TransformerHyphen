import numpy as np

from src.encoding import Encoding
from typing import Any, List
from sklearn.feature_extraction.text import CountVectorizer
from src.utils import remove_hyphenation


class SimpleEmbedding(Encoding):
    """
    Encodes all letters into embeddings of size 32
    """
    def __init__(self, dataset: List[str], unique_letters: List[str]):
        self._letters = unique_letters
        self._letter_encoding = {}
        self._encoding_size = 32
        letter_count = len(self._letters)

        vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3), dtype=np.float32, max_features=32)
        dataset = [remove_hyphenation(word) for word in dataset]
        vectorizer.fit(dataset)

        words_with_letter = {}
        for letter in self._letters:
            words_with_letter[letter] = []

        for word in dataset:
            for letter in self._letters:
                if letter in word:
                    words_with_letter[letter] += word

        for letter in self._letters:
            ngram_features = vectorizer.transform(words_with_letter[letter]).toarray()
            self._letter_encoding[letter] = np.mean(ngram_features, axis=0)  # Aggregate embeddings


    @property
    def letters(self) -> List[str]:
        return self._letters

    @property
    def letter_encoding(self) -> dict[str, List[Any]]:
        return self._letter_encoding

    @property
    def encoding_size(self) -> int:
        return self._encoding_size

class LargerEmbedding(Encoding):
    """
    Encodes all letters into embeddings of size 64

    """
    def __init__(self, dataset: List[str], unique_letters: List[str]):
        self._letters = unique_letters
        self._letter_encoding = {}
        self._encoding_size = 64
        letter_count = len(self._letters)

        vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 5), dtype=np.float32, max_features=self._encoding_size)
        dataset = [remove_hyphenation(word) for word in dataset]
        vectorizer.fit(dataset)

        words_with_letter = {}
        for letter in self._letters:
            words_with_letter[letter] = []

        for word in dataset:
            for letter in self._letters:
                if letter in word:
                    words_with_letter[letter] += word

        for letter in self._letters:
            ngram_features = vectorizer.transform(words_with_letter[letter]).toarray()
            self._letter_encoding[letter] = np.mean(ngram_features, axis=0)  # Aggregate embeddings


    @property
    def letters(self) -> List[str]:
        return self._letters

    @property
    def letter_encoding(self) -> dict[str, List[Any]]:
        return self._letter_encoding

    @property
    def encoding_size(self) -> int:
        return self._encoding_size
