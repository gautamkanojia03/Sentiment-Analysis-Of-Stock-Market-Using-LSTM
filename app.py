from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from util import clean_text  # Importing cleaned_text from main.py

app = Flask(__name__)

# Loading LSTM model
model = load_model('model.h5')

# Loading the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_len = 20  # Defining the maximum length for padding

@app.route('/', methods=['GET', 'POST'])
def home():
    result = {}  # Initializing result as an empty dictionary
    if request.method == 'POST':
        news_headline = request.form['headline']

        # Preprocessing the headline
        input_data = preprocess_text(news_headline)

        # Making prediction
        prediction = model.predict(np.array([input_data]))

        # Processing prediction to get readable output
        sentiment = "buy" if prediction > 0.5 else "sell"
        confidence = prediction[0][0] if sentiment == "buy" else 1 - prediction[0][0]

        sentiment_label = "positive" if prediction > 0.5 else "negative"

        result['stock_action'] = f"We are {confidence * 100:.2f}% sure {sentiment} stock."
        result['sentiment'] = f"The sentiment is {sentiment_label}."

    return render_template('index.html', result=result)

def preprocess_text(text):
    # Cleaning the text using the imported clean_text function
    cleaned_text_result = clean_text(text)

    # Tokenizing the cleaned text
    sequences = tokenizer.texts_to_sequences([cleaned_text_result])

    # Padding sequences to ensure uniform length
    data = pad_sequences(sequences, maxlen=max_len)

    return data[0]  # Returning the preprocessed sequence

if __name__ == '__main__':
    app.run(debug=True)
