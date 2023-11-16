# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 17:58:44 2023

@author: Parth
"""

import streamlit as st
import pickle
from sklearn.svm import SVC
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer



nltk.download('punkt')
nltk.download('stopwords')

# Load the SVM classifier
with open('D:\\data_science\\Projects\\AI Variant\\P304 NLP\\r\\svm_classifier.pkl', 'rb') as f:
    svm_classifier = pickle.load(f)

# Load the TF-IDF vectorizer
with open('D:\\data_science\\Projects\\AI Variant\\P304 NLP\\r\\tfidf_vectorizer.pkl', 'rb') as f:
   vectorizer = pickle.load(f)

# Define the preprocess function
def process_text(text):
    ps = PorterStemmer()
    text = text.lower()
    text = nltk.word_tokenize(text)

    # removing special characters
    y = [i for i in text if i.isalnum()]

    # removing stop words and punctuation
    text = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    # stemming
    text = [ps.stem(i) for i in text]

    return " ".join(text)

# Streamlit app
def main():
    st.title("Email Classifier")

    # Input text box for user to enter an email
    user_email_content = st.text_area("Enter the content of the email:")

    # Check if the user has pressed the "Predict" button
    if st.button("Predict"):
        # Preprocess the user email content
        user_email_transformed = process_text(user_email_content)

        # Vectorize the user email using the loaded TF-IDF vectorizer
        user_email_tfidf = vectorizer.transform([user_email_transformed])

        # Make predictions using the loaded SVM classifier
        user_email_prediction = svm_classifier.predict(user_email_tfidf)

        # Display the predicted class
        result_message = "Non-abusive" if user_email_prediction[0] == 1 else "Abusive"
        st.subheader("Prediction:")
        st.write(f'The predicted class for the user email is: {result_message}')

if __name__ == '__main__':
    main()