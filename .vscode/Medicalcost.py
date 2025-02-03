import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
file_path = "D:/Intern Project/insurance.csv"
df = pd.read_csv(file_path)

# Data Inspection
print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset Information:")
df.info()
print("\nSummary Statistics:")
print(df.describe())

# Checking for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Data Visualization
plt.figure(figsize=(10,5))
sns.histplot(df['charges'], bins=50, kde=True)
plt.title('Distribution of Insurance Charges')
plt.show()

sns.pairplot(df, hue='smoker')
plt.show()

# Additional visualizations to explore relationships
plt.figure(figsize=(10,5))
sns.boxplot(x='smoker', y='charges', data=df)
plt.title('Insurance Charges by Smoking Status')
plt.show()

plt.figure(figsize=(10,5))
sns.boxplot(x='sex', y='charges', data=df)
plt.title('Insurance Charges by Gender')
plt.show()

plt.figure(figsize=(10,5))
sns.boxplot(x='region', y='charges', data=df)
plt.title('Insurance Charges by Region')
plt.show()

plt.figure(figsize=(10,5))
sns.scatterplot(x='bmi', y='charges', hue='smoker', data=df)
plt.title('Insurance Charges vs BMI')
plt.show()

# Encoding categorical variables
encoder = LabelEncoder()
df['sex'] = encoder.fit_transform(df['sex'])
df['smoker'] = encoder.fit_transform(df['smoker'])
df['region'] = encoder.fit_transform(df['region'])

# Feature Selection and Train-Test Split
X = df.drop(columns=['charges'])
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Building (Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Model Performance Metrics:")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-Squared Score: {r2}")

# Predict medical costs for new input
new_data = np.array([[25, 1, 30.5, 0, 0, 2]])  # Example input
new_data_scaled = scaler.transform(new_data)
predicted_cost = model.predict(new_data_scaled)
print(f"Predicted Medical Cost: {predicted_cost[0]}")
