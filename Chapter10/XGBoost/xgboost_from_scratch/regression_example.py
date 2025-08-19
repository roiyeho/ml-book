import time

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from xgb_regressor import XGBRegressor as CustomXGBRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor

# Benchmark on the california housing dataset
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    """Train a regressor, measure training time, and evaluate performance."""
    print(name)
    print('-' * 35)

    # Measure training time
    train_start_time = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - train_start_time
    print(f'Training time: {elapsed:.3f} sec')

    # Evaluate performance
    train_score = model.score(X_train, y_train)
    print(f'R2 score (train): {train_score:.4f}')
    test_score = model.score(X_test, y_test)
    print(f'R2 score (test): {test_score:.4f}\n')

# Define models in a dictionary
models = {
    'CustomXGBRegressor': CustomXGBRegressor(),
    'XGBRegressor': XGBRegressor(random_state=42),
    'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42),
    'HistGradientBoostingRegressor': HistGradientBoostingRegressor(random_state=42),
}

# Evaluate each model and print the results
for name, model in models.items():
    evaluate_model(name, model, X_train, y_train, X_test, y_test)