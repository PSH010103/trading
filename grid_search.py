from models import lr_model
from feature_selection import X_filtered, y
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.tree import DecisionTreeRegressor
import numpy as np

lr_model.fit(X_filtered, y)
y_pred = lr_model.predict(X_filtered)
mape = mean_absolute_percentage_error(y, y_pred)
print(f'Linear Regression MAPE: {mape:.4f}')

from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 9],
    'min_samples_split': [2, 5, 10],
}

grid_search = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=5, scoring='neg_mean_absolute_percentage_error')
grid_search.fit(X_filtered, y)

print(f'Best parameters: {grid_search.best_params_}')
print(f'Best MAPE: {-grid_search.best_score_:.4f}')


from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
mape_scores = []

for train_index, test_index in tscv.split(X_filtered):
    X_train, X_test = X_filtered.iloc[train_index], X_filtered.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = DecisionTreeRegressor(**grid_search.best_params_)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mape_scores.append(mape)

print(f'Expanding Window CV MAPE: {np.mean(mape_scores):.4f}')
