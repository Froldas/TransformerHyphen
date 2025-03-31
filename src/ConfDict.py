from src.models.simple_mlp import *
from src.models.simple_transformer import *

from src.encodings.binary import BinaryEncoding, OneHotEncoding
from src.encodings.embedding import SimpleEmbedding

from src.encodings.float import SimpleFloatEncoding, AdvancedFloatEncoding, AdvancedFloatEncoding2

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
                       "SimpleLargeMLP": SimpleMLP(input_tokens * embedding_size, hidden_size=512, output_size=output_size),
                       "SimpleTransformer": SimpleTransformer(input_tokens, embedding_size, hidden_size=64, output_size=output_size),
                       "SimpleTransformerMasked": SimpleTransformerMasked(input_tokens, embedding_size, hidden_size=64, output_size=output_size),
                       "SimpleTransformerResidual": SimpleTransformerResidual(input_tokens, embedding_size, hidden_size=64, output_size=output_size),
                       "SimpleLargeTransformer": SimpleTransformer(input_tokens, embedding_size, hidden_size=512, output_size=output_size)}


class Encodings:
    """
    All encodings must be implementing src.encodings.encoding interface
    """
    def __init__(self):
        self.encodings = {"binary": BinaryEncoding,
                          "one_hot": OneHotEncoding,
                          "simple_float": SimpleFloatEncoding,
                          "advanced_float": AdvancedFloatEncoding,
                          "advanced_float2": AdvancedFloatEncoding2,
                          "simple_embedding": SimpleEmbedding}
