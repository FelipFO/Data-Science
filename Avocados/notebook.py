# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = 'data.csv'
data = pd.read_csv(file_path)

# Step 1: Exploratory Data Analysis (EDA)

# Basic statistics
print(data.describe())

# Check for missing values
print(data.isna().sum())

# Visualize the distribution of AveragePrice
sns.histplot(data['AveragePrice'], bins=30, kde=True)
plt.title('Distribution of Average Price')
plt.show()

# Convert Date to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Extract numerical features only for the correlation matrix
numeric_data = data.drop(columns=['Date', 'type', 'region'])

# Calculate the correlation matrix
corr_matrix = numeric_data.corr()

# Plot the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Step 2: Data Preprocessing

# Handle missing values
data['SmallBags'].fillna(data['SmallBags'].mean(), inplace=True)
data['LargeBags'].fillna(data['LargeBags'].mean(), inplace=True)
data['XLargeBags'].fillna(data['XLargeBags'].mean(), inplace=True)

# Extract new date features
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Week'] = data['Date'].dt.isocalendar().week
data['DayOfWeek'] = data['Date'].dt.dayofweek

# Drop the Date column as it's now represented in Year, Month, Week, and DayOfWeek
data.drop(columns=['Date'], inplace=True)

# Define categorical and numerical features
cat_features = ['type', 'region']
num_features = data.columns.difference(cat_features + ['AveragePrice']).tolist()

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(), cat_features)
    ])

# Step 3: Model Building

# Split data into features and target
X = data.drop(columns=['AveragePrice'])
y = data['AveragePrice']

# Split data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

# Set up the hyperparameter grid
param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

# Grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='r2')
grid_search.fit(X_train, y_train)

# Get the best parameters and best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(f'Best Parameters: {best_params}')

# Evaluate on validation set
y_val_pred = best_model.predict(X_val)
mse_val = mean_squared_error(y_val, y_val_pred)
r2_val = r2_score(y_val, y_val_pred)

print(f'Validation Mean Squared Error: {mse_val}')
print(f'Validation R^2 Score: {r2_val}')

# Cross-Validation Score
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')
print(f'Cross-Validation R^2 Scores: {cv_scores}')
print(f'Mean Cross-Validation R^2 Score: {np.mean(cv_scores)}')

# Step 4: Evaluation on Test Set

# Predict on test data
y_test_pred = best_model.predict(X_test)

# Calculate metrics
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f'Test Mean Squared Error: {mse_test}')
print(f'Test R^2 Score: {r2_test}')

# Visualize the predictions
plt.scatter(y_test, y_test_pred, alpha=0.3)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Predicted vs True Values')
plt.show()
