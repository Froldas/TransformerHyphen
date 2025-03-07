import sys
from torch import Tensor, load

from src.dataset import HyphenationInterace, insert_hyphenation
from src.model import SimpleMLP

batch_size = 8
data_file = "data/cs-all-cstenten.wlh"

def main():
    hyp_itf = HyphenationInterace.load_configuration()

    loaded_model = SimpleMLP(hyp_itf.input_size, 512, hyp_itf.output_size)
    (loaded_model.load_state_dict
     (load('simple_mlp_model.pth')))
    loaded_model.eval()

    itf = HyphenationInterace.load_configuration()

    data = []
    with open("data/test_text.txt", "r+", encoding="utf-8") as f:
        data = f.readline()

    data = data.split(" ")

    for word in sys.argv[1:]:
    #for word in data:
        input_tensor = Tensor(itf.convert_word_to_input_tensor(word))
        output = loaded_model(input_tensor)
        print(insert_hyphenation(word, output))


if __name__ == "__main__":
    #from cProfile import run
    #run("main()")
    main()