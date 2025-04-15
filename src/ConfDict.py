from src.models.simple_mlp import *
from src.models.simple_transformer import *
from src.models.combined_transformer import *

from src.encodings.binary import BinaryEncoding, AdvancedBinaryEncoding, OneHotEncoding
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
        self.models = {"SimpleMLP": SimpleMLP(input_tokens * embedding_size, hidden_size=64, output_size=output_size),
                       "SimpleLargeMLP": SimpleMLP(input_tokens * embedding_size, hidden_size=512, output_size=output_size),
                       "SimpleTransformer": SimpleTransformer(input_tokens, embedding_size, hidden_size=64, output_size=output_size),
                       "SimpleTransformerResidual": SimpleTransformerResidual(input_tokens, embedding_size, hidden_size=64, output_size=output_size),
                       "SimpleTransformerMaskWindow": SimpleTransformerMaskWindow(input_tokens, embedding_size, hidden_size=64, output_size=output_size),
                       "SimpleTransformerReversed": SimpleTransformerReversed(input_tokens, embedding_size, hidden_size=64, output_size=output_size),
                       "TransformerCombined1": TransformerCombined1(input_tokens, embedding_size, hidden_size=64, output_size=output_size),
                       "SimpleLargeTransformerResidual": SimpleTransformerResidual(input_tokens, embedding_size, hidden_size=4096, output_size=output_size)
                       }


class Encodings:
    """
    All encodings must be implementing src.encodings.encoding interface
    """
    def __init__(self):
        self.encodings = {"binary": BinaryEncoding,
                          "advanced_binary": AdvancedBinaryEncoding,
                          "one_hot": OneHotEncoding,
                          "simple_float": SimpleFloatEncoding,
                          "advanced_float": AdvancedFloatEncoding,
                          "advanced_float2": AdvancedFloatEncoding2,
                          "simple_embedding": SimpleEmbedding}
