import streamlit as st
import xgboost as xgb
import pickle

# Load the data
df = pd.read_csv("data.csv")

# Create a sidebar where users can select the features they want to use
features = df.columns.tolist()
selected_features = st.sidebar.multiselect("Select features", options=features)

# Create a sidebar where users can enter the hyperparameters for the XGBoost model
learning_rate = st.sidebar.slider("Learning rate", 0.01, 0.5, 0.1)
max_depth = st.sidebar.slider("Max depth", 1, 10, 3)
n_estimators = st.sidebar.slider("Number of estimators", 100, 1000, 500)

# Select the target column
target_col = st.text_input("Enter the name of the target column:")

if target_col:
    if target_col in df.columns:
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(df[selected_features], df[target_col], test_size=0.2)

        # Train the XGBoost model
        model = xgb.XGBClassifier(learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators)
        model.fit(X_train, y_train)

        # Evaluate the model
        accuracy = model.score(X_test, y_test)
        st.write(f"Accuracy: {accuracy:.2f}")

        # Create a Pickle file for the model
        pickle.dump(model, open("model.pkl", "wb"))

        # Add a button to download the Pickle file
        if st.button("Download Pickle file"):
            st.markdown("Model saved as `model.pkl` in the current directory.")
            st.download("model.pkl")
    else:
        st.write("The specified target column is not in the dataframe.")
