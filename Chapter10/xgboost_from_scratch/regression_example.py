import time

from sklearn.datasets import make_regression
X, y = make_regression(random_state=0)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

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
    print(f'R2 score (test): {test_score:.4f}')
    print()

print('Benchmarking on the make_regression() data set')
print('-' * 50)

from xgb_regressor import XGBRegressor
my_xgb_reg = XGBRegressor()
evaluate_model('XGBoost (custom)', my_xgb_reg, X_train, y_train, X_test, y_test)

import xgboost
original_xgb_reg = xgboost.XGBRegressor()
evaluate_model('XGBoost (original)', original_xgb_reg, X_train, y_train, X_test, y_test)
 
from sklearn.ensemble import GradientBoostingRegressor
gboost_reg = GradientBoostingRegressor(random_state=0)
evaluate_model('Gradient boosting', gboost_reg, X_train, y_train, X_test, y_test)

from sklearn.ensemble import HistGradientBoostingRegressor
hist_gboost_reg = HistGradientBoostingRegressor(random_state=0)
evaluate_model('Histogram-based gradient boosting', hist_gboost_reg, X_train, y_train, X_test, y_test)

from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
evaluate_model('Linear regression', linear_reg, X_train, y_train, X_test, y_test)

# Benchmark on the california housing data set
print('Benchmarking on the california housing data set')
print('-' * 50)

from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
names = ['XGBoost (custom)', 'XGBoost (original)', 'Gradient boosting', 'Histogram-based gradient boosting', 
         'Linear regression']
models = [XGBRegressor(), 
          xgboost.XGBRegressor(), 
          GradientBoostingRegressor(random_state=0), 
          HistGradientBoostingRegressor(random_state=0),
          LinearRegression()]

for name, model in zip(names, models):
    evaluate_model(name, model, X_train, y_train, X_test, y_test)