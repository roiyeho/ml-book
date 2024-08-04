import time
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from xgb_regressor import XGBRegressor as CustomXGBRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

# Benchmark on the california housing data set
data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    print(name)
    print('-' * 35)
    train_start_time = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - train_start_time
    print(f'Training time: {elapsed:.3f} sec')

    train_score = model.score(X_train ,y_train)
    print(f'R2 score (train): {train_score:.4f}')
    test_score = model.score(X_test, y_test)
    print(f'R2 score (test): {test_score:.4f}\n')

names = ['CustomXGBRegressor', 'XGBRegressor', 'GradientBoostingRegressor', 
         'HistGradientBoostingRegressor']
models = [CustomXGBRegressor(), 
          XGBRegressor(random_state=42), 
          GradientBoostingRegressor(random_state=42), 
          HistGradientBoostingRegressor(random_state=42)]

for name, model in zip(names, models):
    evaluate_model(name, model, X_train, y_train, X_test, y_test)