import numpy as np
from torch import  load
import torch

from src.dataset import HyphenationDataset, HyphenationInterace, insert_hyphenation
from src.model import SimpleMLP


data_file = "data/cs-all-cstenten.wlh"

hyp_itf = HyphenationInterace.load_configuration()

loaded_model = SimpleMLP(hyp_itf.input_size, 512, hyp_itf.output_size)
loaded_model.load_state_dict(load('simple_mlp_model.pth'))
loaded_model.eval()


dataset = HyphenationDataset(data_file=data_file)
X = []
y = []
for data_point in dataset:
    features, label = data_point
    X.append(features)  # Convert features to NumPy array
    y.append(label)

y_pred = loaded_model(torch.Tensor(np.array(X)).to("cpu"))
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(torch.Tensor(np.array(y)).detach().numpy(), y_pred.to("cpu").detach().numpy())

print(f"Full dataset acc: {accuracy}")
with open("wrongly_predicted.txt", "w+", encoding="utf-8") as f:
    for i in range(len(dataset)):
        if not torch.equal(y_pred[i], torch.Tensor(y[i])):
            f.writelines(f"GT: {dataset.words[i]} | PRED: {insert_hyphenation(dataset.words[i].replace("-", ""), y_pred[i])}\n")