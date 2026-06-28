from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, OUTPUT_DIR, MODEL_DIR
from src.data import load_dataset, split_dataset
from src.models import build_baseline_model, build_classical_model, NeuralTextClassifier, save_sklearn_model
from src.evaluate import compute_metrics, save_metrics, save_confusion_matrix, get_classification_report
from src.experiments import preprocessing_experiment


def run_project() -> None:
    """Train, evaluate, and save all project artifacts."""
    df = load_dataset(RAW_DATA_PATH)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    x_train, x_test, y_train, y_test = split_dataset(df)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    models = {
        "baseline_majority": build_baseline_model(),
        "classical_tfidf_logreg": build_classical_model(),
    }
    for name, model in models.items():
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        results[name] = compute_metrics(y_test, preds)
        save_confusion_matrix(y_test, preds, OUTPUT_DIR / f"{name}_confusion_matrix.png", f"{name} Confusion Matrix")
        save_sklearn_model(model, MODEL_DIR / f"{name}.joblib")
        print(f"\n{name}\n{get_classification_report(y_test, preds)}")

    neural = NeuralTextClassifier()
    neural.fit(x_train, y_train)
    neural_preds = neural.predict(x_test)
    results["neural_bow_mlp"] = compute_metrics(y_test, neural_preds)
    save_confusion_matrix(y_test, neural_preds, OUTPUT_DIR / "neural_bow_mlp_confusion_matrix.png", "Neural BOW MLP Confusion Matrix")
    neural.save(MODEL_DIR / "neural_bow_mlp.joblib")
    print(f"\nneural_bow_mlp\n{get_classification_report(y_test, neural_preds)}")

    experiment_df = preprocessing_experiment(x_train, x_test, y_train, y_test)
    experiment_df.to_csv(OUTPUT_DIR / "preprocessing_experiment.csv", index=False)
    save_metrics(results, OUTPUT_DIR / "metrics.json")
    print("\nExperiment results:")
    print(experiment_df.to_string(index=False))


if __name__ == "__main__":
    run_project()
