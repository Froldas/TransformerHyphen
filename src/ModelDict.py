from src.models.simple_mlp import SimpleMLP
from src.models.simple_transformer import SimpleTransformer

class ModelDict:
    def __init__(self, input_tokens, embedding_size, output_size):
        self.models = {"SimpleMLP": SimpleMLP(input_tokens * embedding_size, hidden_size=32, output_size=output_size),
                       "SimpleTransformer": SimpleTransformer(input_tokens, embedding_size, output_size)}
