import sys
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from torch import Tensor, load

from src.dataset import HyphenationInterface
from src.ConfDict import Models
from src.utils import load_yaml_conf, insert_hyphenation


YML_CONF_PATH = "configuration.yml"

attention_weights = []


# Define the hook function
def get_attention_weights(module, input, output):
    # output[1] contains the attention weights
    attention_weights.append(module.attn_weights[0][0].detach().numpy())


def plot_attention(word):
    # Visualize attention map
    ax = sns.heatmap(attention_weights[-1], cmap='viridis')
    plt.title(f'Attention Map - {word}')
    # Set axis labels
    ax.set_xticks(range(len(word)))
    ax.set_yticks(range(len(word)))
    ax.set_xticklabels(word, rotation=90)
    ax.set_yticklabels(word, rotation=0)
    plt.xlabel('Key Positions')
    plt.ylabel('Query Positions')
    plt.show()


def main():
    config = load_yaml_conf(Path(YML_CONF_PATH))
    hyp_itf = HyphenationInterface.load_configuration(config["work_dir"], config["configuration_path"])

    model_path = Path(config["work_dir"]) / config["model_path"]
    loaded_model = Models(hyp_itf.num_input_tokens,
                   hyp_itf.encoding_size,
                   hyp_itf.output_size,
                   config["hyphen_threshold"]).models[config["model"]]
    loaded_model.load_state_dict(load(model_path))
    loaded_model.eval()

    #with open("datasets/test_text.txt", "r+", encoding="utf-8") as f:
    #    data = f.readline()
    #data = data.split(" ")
    # for word in data:

    if config["print_attention_map"]:
        # Register the hook to the desired layer
        # For example, to the first encoder layer's self-attention (cannot guarantee compatibility at all times)
        # Change how desired
        loaded_model.attention.register_forward_hook(get_attention_weights)


    for word in sys.argv[1:]:
        input_tensor = Tensor(hyp_itf.encode(word))
        output = loaded_model(input_tensor)
        if config["print_attention_map"]:
            plot_attention(word)
        print(insert_hyphenation(word, output[0]))


if __name__ == "__main__":
    #from cProfile import run
    #run("main()")
    main()
