import numpy as np

class BagOfWords:
    def __init__(self):
        self.vocabulary_ = {}
        self.inverse_vocabulary_ = []

    def fit(self, documents):
        """Learn the vocabulary from the documents."""
        for document in documents:
            for word in document.split():
                if word not in self.vocabulary_:
                    self.vocabulary_[word] = len(self.vocabulary_)
                    self.inverse_vocabulary_.append(word)
        return self

    def transform(self, documents):
        """Transform documents to document-term matrix."""
        rows = []
        for document in documents:
            row = [0] * len(self.vocabulary_)
            for word in document.split():
                if word in self.vocabulary_:
                    row[self.vocabulary_[word]] += 1
            rows.append(row)
        return rows

    def fit_transform(self, documents):
        """Learn the vocabulary and transform documents to document-term matrix."""
        self.fit(documents)
        return self.transform(documents)
    
    def save(self, path="trained_models/vocab.npy"):
        np.save(path, self.vocabulary_)

    def load(self, path="trained_models/vocab.npy"):
        self.vocabulary_ = np.load(path, allow_pickle=True).item()