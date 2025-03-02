import numpy as np

class BagOfWords:
    def __init__(self):
        self.vocab = {}

    def fit(self, texts):
        unique_words = set()
        for text in texts:
            words = text.lower().split()
            unique_words.update(words)
        self.vocab = {word: i for i, word in enumerate(sorted(unique_words))}

    def transform(self, texts):
        vectors = np.zeros((len(texts), len(self.vocab)))
        for i, text in enumerate(texts):
            words = text.lower().split()
            for word in words:
                if word in self.vocab:
                    vectors[i, self.vocab[word]] += 1
        return vectors

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)

    def save(self, path="models/vocab.npy"):
        np.save(path, self.vocab)

    def load(self, path="models/vocab.npy"):
        self.vocab = np.load(path, allow_pickle=True).item()
