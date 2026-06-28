import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from .evaluate import compute_metrics
from .config import RANDOM_STATE


def preprocessing_experiment(x_train, x_test, y_train, y_test) -> pd.DataFrame:
    """Compare count features vs. TF-IDF features for logistic regression."""
    candidates = {
        "count_unigram": CountVectorizer(ngram_range=(1, 1), max_features=2000),
        "tfidf_unigram": TfidfVectorizer(ngram_range=(1, 1), max_features=2000),
        "tfidf_unigram_bigram": TfidfVectorizer(ngram_range=(1, 2), max_features=2000),
    }
    rows = []
    for name, vectorizer in candidates.items():
        model = Pipeline([
            ("vectorizer", vectorizer),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
        ])
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        row = {"pipeline": name}
        row.update(compute_metrics(y_test, preds))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
