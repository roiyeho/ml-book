import time

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgb_classifier import XGBClassifier as CustomXGBClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

# Benchmark on the breast cancer dataset
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    """Train a classifier, measure the training time, and evaluate performance."""
    print(name)
    print('-' * 35)

    # Measure training time
    train_start_time = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - train_start_time
    print(f'Training time: {elapsed:.3f} sec')

    # Evaluate performance
    train_score = model.score(X_train ,y_train)
    print(f'Accuracy (train): {train_score:.4f}')
    test_score = model.score(X_test, y_test)
    print(f'Accuracy (test): {test_score:.4f}\n')

# Define models in a dictionary
models = {
    'CustomXGBClassifier': CustomXGBClassifier(),
    'XGBClassifier': XGBClassifier(random_state=42),
    'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42),
    'HistGradientBoostingClassifier': HistGradientBoostingClassifier(random_state=42),
}

# Evaluate each model and print the results
for name, model in models.items():
    evaluate_model(name, model, X_train, y_train, X_test, y_test)