from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset and cache it
@st.cache_data
def load_data(data_path):
    return pd.read_csv(data_path)

# Vectorization and model training, cached to speed up execution
@st.cache_resource
def train_model(X_train, y_train):
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    return vectorizer, model

# Load the dataset
data_path = 'Emotion_final_with_predictions.csv'  # Ensure this path is correct for deployment
data = load_data(data_path)

# Prepare data
X = data['Text']
y = data['Emotion']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
vectorizer, logistic_regression_model = train_model(X_train, y_train)

# Make predictions on test set
X_test_vec = vectorizer.transform(X_test)
y_pred = logistic_regression_model.predict(X_test_vec)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
st.write(f'Logistic Regression Accuracy: {accuracy:.4f}')
st.write('Classification Report:')
st.text(classification_report(y_test, y_pred))

# Prediction function
def predict_emotion_logistic(text):
    text_vec = vectorizer.transform([text])
    return logistic_regression_model.predict(text_vec)[0]

# Streamlit app
st.title('Emotion Prediction using Logistic Regression')

# Display the dataset
st.subheader('Dataset')
st.write(data.head())

# Input for prediction
st.subheader('Predict Emotion from Text')
input_text = st.text_input('Enter text:')

if st.button('Predict'):
    if input_text:
        predicted_emotion = predict_emotion_logistic(input_text)
        st.write(f'Predicted Emotion: {predicted_emotion}')
    else:
        st.write('Please enter some text to predict the emotion.')
