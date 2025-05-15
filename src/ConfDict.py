
from src.models.simple_mlp import *
from src.models.simple_transformer import *
from src.models.combined_transformer import *

from src.encodings.binary import BinaryEncoding, AdvancedBinaryEncoding, OneHotEncoding
from src.encodings.embedding import SimpleEmbedding, SimpleEmbedding2, LargerEmbedding
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
    def __init__(self, input_tokens, embedding_size, output_size, hyphen_threshold):
        self.models = {"SimpleMLP": SimpleMLP(input_tokens * embedding_size, hidden_size=128, output_size=output_size, hyphen_threshold=hyphen_threshold),
                       "SimpleDeeperMLP": SimpleDeeperMLP(input_tokens * embedding_size, hidden_size=128, output_size=output_size, hyphen_threshold=hyphen_threshold),
                       "SimpleLargeDeeperMLP": SimpleDeeperMLP(input_tokens * embedding_size, hidden_size=512,
                                                          output_size=output_size, hyphen_threshold=hyphen_threshold),
                       "SimpleMLPConvolution": SimpleMLPConvolution(input_tokens, embedding_size, hidden_size=128, output_size=output_size, hyphen_threshold=hyphen_threshold),
                       "SimpleLargeMLP": SimpleMLP(input_tokens * embedding_size, hidden_size=512, output_size=output_size, hyphen_threshold=hyphen_threshold),
                       "SimpleTransformer": SimpleTransformer(input_tokens, embedding_size, hidden_size=128, output_size=output_size, hyphen_threshold=hyphen_threshold),
                       "SimpleTransformerResidual": SimpleTransformerResidual(input_tokens, embedding_size, hidden_size=128, output_size=output_size, hyphen_threshold=hyphen_threshold),
                       "SimpleTransformerReversed": SimpleTransformerReversed(input_tokens, embedding_size, hidden_size=128, output_size=output_size, hyphen_threshold=hyphen_threshold),
                       "SimpleTransformerConvolution": SimpleTransformerConvolution(input_tokens, embedding_size, hidden_size=128, output_size=output_size, hyphen_threshold=hyphen_threshold),
                       "SimpleTransformerConvolutionSecond": SimpleTransformerConvolution(input_tokens, embedding_size, hidden_size=128, output_size=output_size, hyphen_threshold=hyphen_threshold),
                       "TransformerCombined1": TransformerCombined1(input_tokens, embedding_size, hidden_size=512, output_size=output_size, hyphen_threshold=hyphen_threshold),
                       "TransformerCombined2": TransformerCombined2(input_tokens, embedding_size, hidden_size=512,
                                                                    output_size=output_size,
                                                                    hyphen_threshold=hyphen_threshold),
                       "TransformerCombined3": TransformerCombined3(input_tokens, embedding_size, hidden_size=512,
                                                                    output_size=output_size,
                                                                    hyphen_threshold=hyphen_threshold),
                       "AdvancedTransformerResidualDeep": AdvancedTransformerResidualDeep(input_tokens, embedding_size, hidden_size=128,
                                                                    output_size=output_size,
                                                                    hyphen_threshold=hyphen_threshold),
                       "AdvancedTransformerResidualDeep256": AdvancedTransformerResidualDeep(input_tokens, embedding_size,
                                                                                          hidden_size=256,
                                                                                          output_size=output_size,
                                                                                          hyphen_threshold=hyphen_threshold),
                       "AdvancedTransformerResidualDeepMHead": AdvancedTransformerResidualDeepMHead(input_tokens,
                                                                                             embedding_size,
                                                                                             hidden_size=128,
                                                                                             output_size=output_size,
                                                                                             hyphen_threshold=hyphen_threshold),
                       "SimpleLargeTransformerResidual": SimpleTransformerResidual(input_tokens, embedding_size, hidden_size=512, output_size=output_size, hyphen_threshold=hyphen_threshold)
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
                          "simple_embedding": SimpleEmbedding,
                          "simple_embedding2": SimpleEmbedding2,
                          "large_embedding": LargerEmbedding}
