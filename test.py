import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import string


def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Removing Tags
    text = re.sub('#\S+', '', text).strip()

    # Removing Mentions
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
    
    # words like "not", "no", "very", "but", and "never" can play a crucial role in determining the sentiment
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

def load_test_data(file_path, n_lines=None):
    """Load and preprocess test data in FastText format
    
    Args:
        file_path (str): Path to the test data file
        n_lines (int, optional): Number of lines to load. If None, loads all lines.
    
    Returns:
        pd.DataFrame: DataFrame containing the loaded data with 'label' and 'text' columns
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if n_lines is not None and i >= n_lines:
                break
            # Remove __label__ and the number following it
            text = re.sub(r'__label__\d+\s*', '', line.strip())
            # Extract label from the original line
            label = int(re.search(r'__label__(\d+)', line).group(1))
            data.append({'label': label, 'text': text})
    
    return pd.DataFrame(data)

def evaluate_model(model, tokenizer, test_data, maxlen=100):
    """Evaluate the model on test data"""
    # Clean and preprocess test texts
    cleaned_texts = [clean_text(text) for text in test_data['text']]
    
    # Convert texts to sequences
    sequences = tokenizer.texts_to_sequences(cleaned_texts)
    
    # Pad sequences
    X_test = pad_sequences(sequences, maxlen=maxlen, padding='post')
    
    # Get true labels
    y_true = test_data['label'].values
    
    # Make predictions
    y_pred_proba = model.predict(X_test)
    y_pred = [2 if x > 0.5 else 1 for x in y_pred_proba]
    
    # Calculate metrics with appropriate averaging
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2])
    
    # Print metrics
    print("\nModel Evaluation Metrics:")
    print("-" * 50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    print("\nClassification Report:")
    print("-" * 50)
    print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

def main():
    # Load the model and tokenizer
    print("Loading model and tokenizer...")
    model = load_model('sentiment_model.h5')
    
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    # Load test data (optionally specify number of lines to load)
    print("Loading test data...")
    # test_data = load_test_data('test.ft.txt/test.ft.txt')  # Load all lines
    test_data = load_test_data('test.ft.txt/test.ft.txt', n_lines=10000)
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, tokenizer, test_data)
    
    # Save metrics to file
    with open('evaluation_metrics.txt', 'w') as f:
        f.write("Model Evaluation Metrics:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1']:.4f}\n")
    
    print("\nEvaluation complete! Results saved to 'evaluation_metrics.txt'")
    print("Confusion matrix plot saved as 'confusion_matrix.png'")

if __name__ == "__main__":
    main()
