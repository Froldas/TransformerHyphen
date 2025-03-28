from abc import ABC, abstractmethod
from typing import Any


class Encoding(ABC):
    """
    Abstract Interface which encoding classes must implement
    """

    @property
    @abstractmethod
    def letters(self) -> [str]:
        """
        List of all the letters recognized from the dataset
        This property must be implemented by subclasses
        """
        pass

    @property
    @abstractmethod
    def letter_encoding(self) -> dict[str, [Any]]:
        """
        Dictionary mapping all letters to an input vector
        This property must be implemented by subclasses
        """
        pass

    @property
    @abstractmethod
    def encoding_size(self) -> int:
        """
        Size of the encoding per letter
        This property must be implemented by subclasses
        """
        pass
