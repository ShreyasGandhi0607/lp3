# Importing required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# Load the dataset
data = pd.read_csv('uber.csv')

# Initial Exploration
print("Dataset Head:\n", data.head())
print("\nDataset Tail:\n", data.tail())
print("\nDataset Info:\n")
data.info()
print("\nDataset Description:\n", data.describe())
print("\nDataset Shape:", data.shape)
print("\nDataset Columns:", data.columns)

# Drop unnecessary columns
data = data.drop(['Unnamed: 0', 'key'], axis=1)
print("\nShape after dropping unnecessary columns:", data.shape)

# Extract month and hour from pickup_datetime
data['month'] = data['pickup_datetime'].str.slice(start=5, stop=7)
data['hour'] = data['pickup_datetime'].str.slice(start=11, stop=13)
data = data.drop(['pickup_datetime'], axis=1)
print("\nDataset after extracting month and hour:\n", data.head())

# Handle missing values by replacing with column mean
data["dropoff_longitude"] = data["dropoff_longitude"].fillna(data['dropoff_longitude'].mean())
data["dropoff_latitude"] = data["dropoff_latitude"].fillna(data['dropoff_latitude'].mean())

# Calculate distance using the haversine formula
def haversine(lon1, lon2, lat1, lat2):
    lon1, lon2, lat1, lat2 = map(np.radians, [lon1, lon2, lat1, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return 2 * 6371 * np.arcsin(np.sqrt(a))

data['distance'] = haversine(data['pickup_longitude'], data['dropoff_longitude'],
                             data['pickup_latitude'], data['dropoff_latitude'])

# Replace zeros with the mean for specific columns
data['passenger_count'] = data['passenger_count'].replace(0, data['passenger_count'].mean())
data['distance'] = data['distance'].replace(0, data['distance'].mean())
data.loc[data['fare_amount'] <= 0, 'fare_amount'] = data['fare_amount'].mean()

# Handle any remaining NaN values in the distance column
data['distance'].fillna(data['distance'].mean(), inplace=True)

# Data Visualization - Individual Box Plots for Outlier Detection
columns_to_plot = ['fare_amount', 'pickup_longitude', 'pickup_latitude', 
                   'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'distance']

plt.figure(figsize=(16, 12))
for i, col in enumerate(columns_to_plot, 1):
    plt.subplot(4, 2, i)  # Adjust layout to have fewer plots per row
    sns.boxplot(data[col])
    plt.title(f"Boxplot of {col}")

plt.tight_layout()  # Prevent overlapping of subplots
plt.show()

# Outlier Handling using IQR
def remove_outlier(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.clip(df[col], lower_bound, upper_bound)
    return df

for column in ['fare_amount', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'distance']:
    data = remove_outlier(data, column)

# Plot after outlier treatment
plt.figure(figsize=(16, 12))
for i, col in enumerate(columns_to_plot, 1):
    plt.subplot(4, 2, i)
    sns.boxplot(data[col])
    plt.title(f"Boxplot of {col} after Outlier Treatment")

plt.tight_layout()
plt.show()

# Check for remaining missing values
print("\nMissing Values after imputation:\n", data.isnull().sum())

# Correlation Heatmap
corr = data.corr()
fig, axis = plt.subplots(figsize=(12, 8))
sns.heatmap(corr, annot=True)
plt.show()

# Split dataset into features and target variable
X = data.drop(columns=['fare_amount'])
y = data['fare_amount']
print("\nFeature and Target Shapes:", X.shape, y.shape)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
print("Train and Test Split Shapes:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Linear Regression Model
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)
y_pred_lr = linear_regression.predict(X_test)

# Evaluation Metrics for Linear Regression
print("\nLinear Regression Performance:")
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred_lr))
print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred_lr))
print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, y_pred_lr)))
print("R Squared (R²):", metrics.r2_score(y_test, y_pred_lr))

# Random Forest Regression Model
random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)

# Evaluation Metrics for Random Forest
print("\nRandom Forest Regression Performance:")
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred_rf))
print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred_rf))
print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf)))
print("R Squared (R²):", metrics.r2_score(y_test, y_pred_rf))
