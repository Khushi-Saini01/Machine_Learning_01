# app.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Page setup
st.set_page_config(page_title="Study Hours vs Scores Predictor", layout="centered")
st.title("Study Hours vs Exam Scores Predictor")

# Load the dataset
import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    df = pd.read_csv("hours_vs_scores_100.csv")  
    return df

dataset = load_data()
st.dataframe(dataset)

st.subheader("Dataset Preview")
st.dataframe(dataset.head())

# EDA Section
if st.checkbox("Show Dataset Info"):
    buffer = []
    dataset.info(buf=buffer)
    info_str = "\n".join(buffer)
    st.text(info_str)

if st.checkbox("Show Missing Values"):
    st.write(dataset.isnull().sum())

if st.checkbox("Show Descriptive Statistics"):
    st.write(dataset.describe())

# Visualizations
st.subheader("Visualizations")

if st.checkbox("Show Boxplot of Hours"):
    fig1, ax1 = plt.subplots()
    sns.boxplot(x=dataset["Hours"], ax=ax1)
    ax1.set_title("Boxplot of Study Hours")
    st.pyplot(fig1)

if st.checkbox("Show Distribution Plot of Hours"):
    fig2 = sns.displot(x=dataset["Hours"], kde=True)
    st.pyplot(fig2)

if st.checkbox("Show Scatter Plot of Hours vs Scores"):
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x="Hours", y="Scores", data=dataset, ax=ax3)
    ax3.set_title("Hours vs Scores")
    st.pyplot(fig3)

# Splitting & Training
X = dataset[["Hours"]]
y = dataset["Scores"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

# Show coefficients
st.subheader("Model Parameters")
st.write(f"Coefficient (Slope): {lr.coef_[0]:.4f}")
st.write(f"Intercept: {lr.intercept_:.4f}")

# Plot regression line
y_pred_full = lr.predict(dataset[["Hours"]])
fig4, ax4 = plt.subplots()
sns.scatterplot(x="Hours", y="Scores", data=dataset, ax=ax4, label="Actual Data")
ax4.plot(dataset["Hours"], y_pred_full, color='red', label="Regression Line")
ax4.legend()
st.pyplot(fig4)

# Model Evaluation
y_pred_test = lr.predict(X_test)
st.subheader(" Model Evaluation on Test Set")
st.write(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred_test):.2f}")
st.write(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred_test):.2f}")
st.write(f"R² Score: {r2_score(y_test, y_pred_test) * 100:.2f}%")
st.write(f"Accuracy: {lr.score(X_test, y_test) * 100:.2f}%")

# Actual vs Predicted Plot
fig5, ax5 = plt.subplots()
ax5.scatter(y_test, y_pred_test)
ax5.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
ax5.set_xlabel("Actual Scores")
ax5.set_ylabel("Predicted Scores")
ax5.set_title("Actual vs Predicted Scores")
st.pyplot(fig5)

# Predict user input
st.subheader("Try Predicting")
user_hours = st.slider("Select Study Hours", 0.0, 10.0, 5.0, 0.5)
predicted_score = lr.predict([[user_hours]])
st.success(f"If you study for {user_hours} hours, you may score approximately {predicted_score[0]:.2f} marks.")
