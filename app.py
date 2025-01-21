import streamlit as st
import base64
import joblib
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from googletrans import Translator
from nltk.corpus import stopwords
import nltk

# Download necessary nltk resources
nltk.download('stopwords')
nltk.download('punkt')

# Load the Random Forest model
rf_model = joblib.load('random_forest_text_classifier.joblib')

# Load the BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Use the correct tokenizer path if you're using a custom BERT model
bert_model = BertForSequenceClassification.from_pretrained('saved_bert_model') 

# Ensure the model is in evaluation mode
bert_model.eval()

# Function to encode the image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Apply custom CSS for styling
def inline_css():
    # Get the Base64 image
    image_path = "images/martin-martz-7ELYu7jeEwo-unsplash.jpg"  # Ensure this path is correct
    base64_image = get_base64_image(image_path)

    # Apply the background image through Base64
    st.markdown(f"""
    <style>
    /* Background image */
    .stApp {{
        background: url('data:image/jpeg;base64,{base64_image}') no-repeat center center fixed;
        background-size: cover;
        color: white;
    }}
    /* Title styling */
    h1 {{
        color: white;  /* Make title text white */
        text-align: center;
    }}
    /* Text area styling */
    .stTextArea > div > div > textarea {{
        border-radius: 10px;
        padding: 10px;
        border: 1px solid #ccc;
        font-size: 16px;
    }}
    /* Radio button styling */
    .stRadio > div > div > label {{
        font-size: 16px;
        color: #333;
    }}
    /* Submit button styling */
    .stButton > button {{
        background-color: #4B8BBE;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }}
    .stButton > button:hover {{
        background-color: #36799f;
    }}
    /* Divider styling */
    hr {{
        border: 0;
        height: 1px;
        background: #ccc;
        margin: 20px 0;
    }}
    /* Prediction Result Highlight */
    .prediction-result {{
        background-color: #4CAF50;  /* Green background */
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        margin-top: 20px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
    }}
    </style>
    """, unsafe_allow_html=True)

# Function to clean text (same as used during training)
stop_words = set(stopwords.words('english'))  # English stopwords
def clean_text(text):
    # Tokenize and remove stopwords
    word_tokens = word_tokenize(text.lower())
    cleaned_text = [word for word in word_tokens if word.isalnum() and word not in stop_words]
    return ' '.join(cleaned_text)

# Function for Random Forest prediction
def rf_predict_text(user_input, language):
    # If the text is in Arabic, translate it to English
    if language == 'Arabic':
        translator = Translator()
        user_input = translator.translate(user_input, src='ar', dest='en').text

    # Clean the input text using the same cleaning function that was used during training
    cleaned_text = clean_text(user_input)

    # Get the prediction from the Random Forest model
    prediction = rf_model.predict([cleaned_text])

    # Decode the prediction (0: Not Generated, 1: Generated)
    label = "AI Generated" if prediction[0] == 1 else "Human Generated"
    
    return label

# Function for BERT prediction
def bert_predict_text(user_input, language):
    # If the text is in Arabic, translate it to English
    if language == 'Arabic':
        translator = Translator()
        user_input = translator.translate(user_input, src='ar', dest='en').text

    # Clean the input text using the same cleaning function that was used during training
    cleaned_text = clean_text(user_input)

    # Tokenize and prepare the input text for BERT
    inputs = tokenizer(cleaned_text, return_tensors='pt', truncation=True, padding=True, max_length=512)

    # Get the prediction from the BERT model
    with torch.no_grad():
        outputs = bert_model(**inputs)
        logits = outputs.logits

    # Get the predicted label (0: Not Generated, 1: Generated)
    prediction = torch.argmax(logits, dim=-1).item()

    # Decode the prediction
    label = "AI Generated" if prediction == 1 else "Human Generated"
    
    return label

# Set the page configuration
st.set_page_config(
    page_title="AI vs Human: Academic Essay Authenticity Challenge",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="auto",
)

inline_css()

# Header with inline styling
st.markdown("<h1 style='color: white;'>📝AI vs Human: Academic Essay Authenticity Challenge</h1>", unsafe_allow_html=True)

# Create a form
with st.form(key='text_form'):
    # Rectangle Text Area
    user_input = st.text_area("Enter your text here:", height=200)

    # Language Selection
    language = st.radio(
        "Select Language:",
        ('English', 'Arabic'),
        horizontal=True
    )

    # Model Selection
    model_type = st.radio(
        "Select Model:",
        ('Random Forest', 'BERT'),
        horizontal=True
    )

    # Submit Button
    submit_button = st.form_submit_button(label='Submit')

# Handle form submission and prediction
if submit_button:
    if user_input.strip() == "":
        st.error("Please enter some text before submitting.")
    else:
        # Call the backend prediction function based on the selected model
        if model_type == 'Random Forest':
            result = rf_predict_text(user_input, language)
        else:
            result = bert_predict_text(user_input, language)

        # Display the text entered and the prediction result
        st.markdown(f"**You entered:** {user_input}")
        st.markdown(f"**Selected Language:** {language}")
        st.markdown(f"**Selected Model:** {model_type}")
        
        # Display the result with highlighted background
        st.markdown(f'<div class="prediction-result">Prediction Result: {result}</div>', unsafe_allow_html=True)

        # Display the text in the selected language direction
        if language == 'Arabic':
            st.markdown(f"<div style='direction: rtl; text-align: right;'>{user_input}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='direction: ltr; text-align: left;'>{user_input}</div>", unsafe_allow_html=True)
