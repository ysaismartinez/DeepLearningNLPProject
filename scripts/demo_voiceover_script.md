# Demo Voiceover Script

This project is an NLP ticket triage system. The goal is to read a short support ticket and classify it as urgent or non-urgent so a service team can prioritize risky customer situations faster.

The repository includes three required modeling approaches. First, there is a naive majority-class baseline. Second, there is a classical machine learning model using TF-IDF features and logistic regression. Third, there is a neural network model implemented in PyTorch over bag-of-words features.

I also included a focused experiment comparing different text preprocessing pipelines: count features, TF-IDF unigrams, and TF-IDF unigrams plus bigrams. The purpose is to test whether the representation of the text changes performance.

To run the project, install the requirements and run python main.py. The script loads the dataset, cleans the text, creates a stratified train-test split, trains all three models, saves metrics, saves confusion matrices, and writes the experiment results.

The best model in this run was the classical TF-IDF logistic regression model. It achieved strong overall accuracy and macro F1, while keeping urgent-ticket precision high. The main limitation is that the dataset is small, so this should be treated as a prototype rather than a production-ready system.

For future work, I would collect real support tickets, add more labels, test transformer-based models, and build a human-in-the-loop review flow for borderline predictions.
