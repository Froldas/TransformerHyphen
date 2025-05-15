import numpy as np
import torch
import torch.nn as nn

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

        vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(1, 3), dtype=np.float32, max_features=32)
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


class SimpleEmbedding2(Encoding):
    """
    Encodes all letters into embeddings of size 32
    """

    def __init__(self, dataset: List[str], unique_letters: List[str]):
        self._letters = unique_letters
        self._letter_encoding = {}
        self._encoding_size = 32
        letter_count = len(self._letters)


        dataset = [remove_hyphenation(word) for word in dataset]

        words_with_letter = {}

        char_to_idx = {char: idx for idx, char in enumerate(self._letters + ['<unk>', '<pad>'] )}
        idx_to_char = {idx: char for char, idx in char_to_idx.items()}

        # Step 2: Convert strings to sequences of indices
        sequences = []
        for s in dataset:
            seq = [char_to_idx.get(char, char_to_idx['<unk>']) for char in s]
            sequences.append(seq)

        # Step 3: Pad sequences to the same length
        # Find the length of the longest sequence
        max_len = max(len(seq) for seq in sequences)
        # Pad sequences with the index of '<pad>' token
        padded_sequences = [seq + [char_to_idx['<pad>']] * (max_len - len(seq)) for seq in sequences]
        # Convert to a tensor
        input_tensor = torch.tensor(padded_sequences, dtype=torch.long)  # Shape: (batch_size, seq_len)

        # Step 4: Create an embedding layer
        vocab_size = len(char_to_idx)
        embedding_dim = 32  # You can choose any embedding dimension
        embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        embedded = embedding_layer(input_tensor)

        for letter in self._letters:
            words_with_letter[letter] = []

        for word in dataset:
            for letter in self._letters:
                if letter in word:
                    words_with_letter[letter] += word

        char_embeddings = {char: embedding_layer(torch.tensor([idx])) for char, idx in char_to_idx.items()}

        for char, idx in char_to_idx.items():
            self._letter_encoding[char] = embedding_layer(torch.tensor([idx])).detach().numpy()[0]

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

        vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 5), dtype=np.float32,
                                     max_features=self._encoding_size)
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
