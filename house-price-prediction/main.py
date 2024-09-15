import pandas as pd
from sklearn.model_selection import ShuffleSplit, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import joblib

# Load and preprocess data
data = pd.read_csv('house_prices.csv')

# Separate features (X) and target variable (y)
X = data.drop('price', axis=1)
y = data['price']

# Print dataset size
print(f"Dataset size: {len(data)} samples")

# Create a pipeline with StandardScaler and RandomForestRegressor
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Feature scaling
    ('model', RandomForestRegressor(random_state=42))
])

# Set up hyperparameter grid for RandomForest
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [None, 5, 10],
    'model__min_samples_split': [2, 5],
    'model__min_samples_leaf': [1, 2]
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit the GridSearchCV model
grid_search.fit(X, y)

# Get the best model
best_model = grid_search.best_estimator_

# Output the best hyperparameters
print(f"Best Hyperparameters: {grid_search.best_params_}")

# Use ShuffleSplit for random splitting
ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

# Perform cross-validation using ShuffleSplit
cv_scores = cross_val_score(best_model, X, y, cv=ss, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)

# Print the cross-validation results
print(f"Cross-Validation RMSE scores: {cv_rmse}")
print(f"Mean Cross-Validation RMSE: {np.mean(cv_rmse)}")

# Train the best model on the entire dataset
best_model.fit(X, y)

# Save the trained pipeline (including the scaler and tuned model)
joblib.dump(best_model, 'model_pipeline.pkl')

# Extract the RandomForest model to get feature importances
model = best_model.named_steps['model']
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
print("\nFeature Importances:")
print(feature_importance.sort_values('importance', ascending=False))
