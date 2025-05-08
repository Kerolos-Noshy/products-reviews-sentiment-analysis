import streamlit as st
import re
import string
import pickle

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Function to clean text
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove Tags
    text = re.sub('#\S+', '', text).strip()

    # Remove Mentions
    text = re.sub('@\S+', '', text).strip()

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    
    # Words that could play a crucial role in sentiment analysis
    words_to_remove = ["not", "no", "never", "neither", "nor", "very", 
                       "really", "too", "extremely", "quite", "but", "however", 
                       "although", "though", "if", "unless", "except"]

    stop_words = [word for word in stop_words if word not in words_to_remove]

    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back into text
    cleaned_text = ' '.join(tokens)
    
    # Remove extra whitespace
    cleaned_text = ' '.join(cleaned_text.split())
    
    return cleaned_text

# Load the trained model and tokenizer
model = load_model('sentiment_model.h5')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Streamlit UI components
st.title("Sentiment Analysis App")

# Input text field for user to enter a sentence
text = st.text_area("Enter text for sentiment analysis:", "this product is amazing")

# Clean the input text
new_text = [clean_text(text)]
seq = tokenizer.texts_to_sequences(new_text)
padded = pad_sequences(seq, maxlen=100, padding='post')

# Predict sentiment
if st.button("Predict Sentiment"):
    pred = model.predict(padded)
    
    if pred[0][0] > 0.5:
        st.success(f"Sentiment: Positive {pred[0][0] * 100 :.2f}%")
    else:
        st.error(f"Sentiment: Negative {pred[0][0] * 100 :.2f}%")

# when i buy it i think it is amazing but it broken after 2 days and i recognized that it is scam, too bad product
# this product isn't bad, it is amazing and i really recommend it