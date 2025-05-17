# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
dataset = pd.read_csv("hours_vs_scores_100.csv")

# Display first few rows
print(" First 5 rows of dataset:")
print(dataset.head())

# Dataset info
print("\n Dataset Information:")
print(dataset.info())

# Missing values
print("\n Missing Values:")
print(dataset.isnull().sum())

# Descriptive stats
print("\n Statistical Summary:")
print(dataset.describe())

# Boxplot of Hours
sns.boxplot(x=dataset["Hours"])
plt.title("Boxplot of Study Hours")
plt.show()

# Distribution plot
sns.displot(x=dataset["Hours"], kde=True)
plt.title("Distribution of Study Hours")
plt.show()

# Scatter plot of Hours vs Scores
plt.figure(figsize=(5, 3))
sns.scatterplot(x="Hours", y="Scores", data=dataset)
plt.title("Hours vs Scores")
plt.xlabel("Hours Studied")
plt.ylabel("Scores Obtained")
plt.show()

# Splitting the dataset
X = dataset[["Hours"]]
y = dataset["Scores"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Coefficient and Intercept
print("\nModel Parameters:")
print("Coefficient (m):", lr.coef_[0])
print("Intercept (c):", lr.intercept_)

# Predict on full dataset for plotting
y_full_pred = lr.predict(dataset[["Hours"]])

# Regression Line Plot
sns.scatterplot(x="Hours", y="Scores", data=dataset, label="Actual Data")
plt.plot(dataset[["Hours"]], y_full_pred, color="red", label="Prediction Line")
plt.title("Regression Line: Hours vs Scores")
plt.legend()
plt.show()

# Predict on test set
y_pred = lr.predict(X_test)

# Evaluation Metrics
print("\nEvaluation on Test Set:")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred) * 100, "%")

# Model score
accuracy = lr.score(X_test, y_test)
print("Model Accuracy (R²):", accuracy * 100, "%")

# Plot: Actual vs Predicted
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")
plt.title("Actual vs Predicted Scores")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # 45° line
plt.show()
