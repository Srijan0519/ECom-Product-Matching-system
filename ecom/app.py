import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
import joblib

# Load your trained models and scaler
clf_amazon = joblib.load('clf_amazon.joblib')
clf_instasport = joblib.load('clf_instasport.joblib')
scaler = joblib.load('scaler.joblib')

# Load the sentence transformer model
@st.cache(allow_output_mutation=True)
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

model = load_model()

def predict_availability(title):
    embedding = model.encode([title])
    embedding_scaled = scaler.transform(embedding)
    
    amazon_prob = clf_amazon.predict_proba(embedding_scaled)[0][1]
    instasport_prob = clf_instasport.predict_proba(embedding_scaled)[0][1]
    
    avg_confidence = (amazon_prob + instasport_prob) / 2
    
    return amazon_prob, instasport_prob, avg_confidence

def get_availability_status(probability):
    if probability > 0.7:
        return "Highly likely to be available"
    elif probability > 0.3:
        return "May be available"
    else:
        return "Unlikely to be available"

# Streamlit app
st.title('Product Availability Predictor')

user_input = st.text_input("Enter a product name:")

if st.button('Predict Availability'):
    if user_input:
        amazon_prob, instasport_prob, avg_confidence = predict_availability(user_input)
        
        st.write(f"Predictions for '{user_input}':")
        st.write(f"Amazon: {amazon_prob:.2%} - {get_availability_status(amazon_prob)}")
        st.write(f"Instasport: {instasport_prob:.2%} - {get_availability_status(instasport_prob)}")
        st.write(f"Average confidence: {avg_confidence:.2%}")
    else:
        st.write("Please enter a product name.")