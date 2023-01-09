import streamlit as st
import xgboost as xgb
import pandas as pd

# Load the data
@st.cache
def load_data(file):
    data = pd.read_csv(file)
    return data
#data = pd.read_csv("my_data.csv")

# Preprocess the data
data = data.dropna()
data = pd.get_dummies(data)

# Split the data into training and test sets
train_x, test_x, train_y, test_y = train_test_split(data.drop(columns=["target"]), data["target"], test_size=0.2)

# Train an XGBoost model
model = xgb.XGBClassifier()
model.fit(train_x, train_y)

# Evaluate the model
accuracy = model.score(test_x, test_y)
st.write("Accuracy: ", accuracy)

# Use a slider to input custom parameters for the model
learning_rate = st.slider("Learning rate", 0.01, 0.5, 0.1)
n_estimators = st.slider("Number of estimators", 10, 500, 100)
max_depth = st.slider("Max depth", 2, 10, 3)

# Re-train the model with the custom parameters
model = xgb.XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)
model.fit(train_x, train_y)

# Re-evaluate the model with the custom parameters
accuracy = model.score(test_x, test_y)
st.write("Accuracy with custom parameters: ", accuracy)

# Use a dropdown menu to select a column to make predictions on
columns = st.multiselect("Select a column", data.columns, default=data.columns[0])
prediction = model.predict(data[columns])
st.write("Prediction: ", prediction)
