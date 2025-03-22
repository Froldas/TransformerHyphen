from abc import ABC, abstractmethod
from typing import Any


class Encoding(ABC):
    """
    Abstract Interface which encoding classes must implement
    """
    @property
    @abstractmethod
    def letters(self) -> [str]:
        """This must be implemented by subclasses"""
        pass

    @property
    @abstractmethod
    def letter_encoding(self) -> dict[str, [Any]]:
        """This must be implemented by subclasses"""
        pass

    @property
    @abstractmethod
    def encoding_size(self) -> int:
        """This must be implemented by subclasses"""
        pass