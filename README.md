# NLP Ticket Triage Project

This project classifies customer support tickets as `urgent` or `non_urgent` using natural language processing.

## Project structure

- `data/raw/support_tickets.csv`: original labeled dataset
- `data/processed/`: generated cleaned dataset
- `src/data.py`: loading, validation, and preprocessing
- `src/models.py`: baseline, classical ML, and neural network models
- `src/evaluate.py`: metrics and confusion matrices
- `src/experiments.py`: preprocessing comparison experiment
- `main.py`: trains and evaluates all models
- `outputs/`: generated metrics, experiment results, and confusion matrices
- `reports/`: final technical report

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Models

1. Naive baseline: majority class classifier.
2. Classical ML: TF-IDF with logistic regression.
3. Deep learning: PyTorch feed-forward neural network over bag-of-words features.

## Focused experiment

The experiment compares CountVectorizer, TF-IDF unigram, and TF-IDF unigram + bigram pipelines to test whether feature preprocessing materially affects urgent-ticket detection.
