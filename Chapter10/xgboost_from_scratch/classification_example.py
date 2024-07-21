import time

from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgb_classifier import XGBClassifier
import xgboost

from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, class_sep=0.1, random_state=0)

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
    print(f'Accuracy (train): {train_score:.4f}')
    test_score = model.score(X_test, y_test)
    print(f'Accuracy (test): {test_score:.4f}')
    print()

names = ['XGBoost (custom)', 'XGBoost (original)', 'Gradient boosting', 'Histogram-based gradient boosting', 
         'Logistic regression']
models = [XGBClassifier(), 
          xgboost.XGBClassifier(), 
          GradientBoostingClassifier(random_state=0), 
          HistGradientBoostingClassifier(random_state=0),
          LogisticRegression()]

for name, model in zip(names, models):
    evaluate_model(name, model, X_train, y_train, X_test, y_test)

