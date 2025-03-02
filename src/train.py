import numpy as np
from datasets import load_dataset
from models.logistic_regression import LogisticRegression
from models.bag_of_words import BagOfWords

import os
if not os.path.exists("datasets"):
    os.makedirs("datasets")

dataset = load_dataset("dmitva/human_ai_generated_text", split="train", streaming=True)

def get_batch(dataset, batch_size=5000):
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
X_train = bow.fit_transform(train_texts)

model = LogisticRegression(lr=0.01, epochs=1000)
model.fit(X_train, train_labels)

model.save("models/logistic_weights.npz")
bow.save("models/vocab.npy")

print("Treinamento concluído. Modelo salvo!")
