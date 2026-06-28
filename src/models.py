import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from .config import RANDOM_STATE


def build_baseline_model() -> DummyClassifier:
    """Majority-class baseline."""
    return DummyClassifier(strategy="most_frequent")


def build_classical_model() -> Pipeline:
    """TF-IDF plus logistic regression classifier."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=2000)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])


class SimpleTextNN(nn.Module):
    """A compact neural network over bag-of-words vectors."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, features):
        return self.network(features)


class NeuralTextClassifier:
    """Count-vectorized text classifier trained with PyTorch."""
    def __init__(self, max_features: int = 1000, epochs: int = 20, learning_rate: float = 0.01):
        self.vectorizer = CountVectorizer(max_features=max_features, ngram_range=(1, 2))
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model = None

    def fit(self, texts, labels):
        x = self.vectorizer.fit_transform(texts).astype(np.float32).toarray()
        y = np.asarray(labels, dtype=np.int64)
        torch.manual_seed(RANDOM_STATE)
        self.model = SimpleTextNN(input_dim=x.shape[1])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        dataset = TensorDataset(torch.tensor(x), torch.tensor(y))
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        self.model.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                loss = loss_fn(self.model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, texts):
        if self.model is None:
            raise ValueError("Model has not been fit.")
        x = self.vectorizer.transform(texts).astype(np.float32).toarray()
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(x))
            return torch.argmax(logits, dim=1).numpy()

    def save(self, path):
        joblib.dump({"vectorizer": self.vectorizer, "state_dict": self.model.state_dict(), "input_dim": len(self.vectorizer.get_feature_names_out())}, path)


def save_sklearn_model(model, path):
    joblib.dump(model, path)
