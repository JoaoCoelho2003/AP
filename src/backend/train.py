import numpy as np
import sys
import os
from datasets import load_dataset
from models.logistic_regression import LogisticRegression
from models.dnn.neuralnet import NeuralNetwork
from models.dnn.layers import DenseLayer
from models.dnn.metrics import mse, accuracy
from models.bag_of_words import BagOfWords
from models.dnn.activation import SigmoidActivation, ReLUActivation
from models.dnn.losses import BinaryCrossEntropy

class DatasetWrapper:
    def __init__(self, X, y):
        self.X = X
        self.y = y

dataset = load_dataset("dmitva/human_ai_generated_text", split="train", streaming=True)

def get_batch(dataset, batch_size=6000):
    texts, labels = [], []
    for i, example in enumerate(dataset):
        if "human_text" in example and "ai_text" in example:
            texts.append(example["human_text"])
            labels.append(0) 
            texts.append(example["ai_text"])
            labels.append(1)
        if len(texts) >= batch_size:
            break
    return texts, np.array(labels)

train_texts, train_labels = get_batch(dataset)

bow = BagOfWords()
X_train = np.array(bow.fit_transform(train_texts))
model_type = sys.argv[1] if len(sys.argv) > 1 else "logistic"

if model_type == "neuralnet":
    model = NeuralNetwork(epochs=50, batch_size=32, learning_rate=0.0001, verbose=True,
                        loss=BinaryCrossEntropy, metric=accuracy)
    
    n_features = X_train.shape[1]
    model.add(DenseLayer(32, (n_features,)))
    model.add(ReLUActivation())
    
    model.add(DenseLayer(16))
    model.add(ReLUActivation())
    model.add(DenseLayer(1)) 
    model.add(SigmoidActivation())
    
    dataset = DatasetWrapper(X_train, train_labels)
    model.fit(dataset)
else:
    model = LogisticRegression(lr=0.01, epochs=300)
    model.fit(X_train, train_labels)

if not os.path.exists("trained_models"):
    os.makedirs("trained_models")

if model_type == "neuralnet":
    model.save("trained_models/neuralnet_weights.npz")
    bow.save("trained_models/vocab_dnn.npy")
else:
    model.save("trained_models/logistic_weights.npz")
    bow.save("trained_models/vocab_log.npy")

print("Training completed. Model saved!")
