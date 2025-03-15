from src.models.simple_mlp import SimpleMLP
from src.models.simple_transformer import SimpleTransformer

from src.encodings.binary import BinaryEncoding
#from src.encodings.word_embeddings import SimpleTransformer
#from src.encodings.float import float

"""
File tracking the mapping of the configuration fields:
   * model
   * encoding
"""


class Models:
    """
    All models must inherit from torch.nn.Module
    """
    def __init__(self, input_tokens, embedding_size, output_size):
        self.models = {"SimpleMLP": SimpleMLP(input_tokens * embedding_size, hidden_size=32, output_size=output_size),
                       "SimpleTransformer": SimpleTransformer(input_tokens, embedding_size, hidden_size=32, output_size=output_size)}


class Encodings:
    """
    All encodings must be implementing src.encodings.encoding interface
    """
    def __init__(self):
        self.encodings = {"binary": BinaryEncoding,
                          "word_embeddings": True,
                          "float": True}
