import pandas as pd
import numpy as np

df = pd.read_csv('sentiment_news.csv',encoding='ISO-8859-1')
print(df.head())
print(len(df))

# Shuffling the dataframe
shuffled_df = df.sample(frac=1.0, random_state=42)

# Resetting index
shuffled_df.reset_index(drop=True, inplace=True)
print(shuffled_df.head())

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from util import clean_text

# Applying the clean_text function to the 'text' column, handling NaN values
shuffled_df['cleaned_text'] = shuffled_df['news'].apply(lambda x: clean_text(str(x)) if pd.notna(x) else '')

# cleaned DataFrame
print(shuffled_df.iloc[24])


shuffled_df.drop(24, inplace=True)

texts = shuffled_df['cleaned_text']

labels = np.array(shuffled_df.iloc[:,-2])

# Tokenizing the text data
from keras.preprocessing.text import Tokenizer
max_words = 1000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Padding sequences to ensure uniform length
from keras.utils import pad_sequences
max_len = 20
data = pad_sequences(sequences, maxlen=max_len)

print(data)

len(data),len(labels)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Splitting the data into training and test sets (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

from keras.layers import LSTM, Dropout, Embedding, Dense

# LSTM model
from keras import Sequential
from keras.layers import Embedding,Dense
from tensorflow.keras.layers import LSTM, Dropout


model = Sequential()
model.add(Embedding(max_words, 100, input_length=max_len))  # Increase embedding dimensionality for better representation
model.add(LSTM(128, return_sequences=True))  # Increase the number of LSTM units and add return_sequences=True to stack LSTM layers
model.add(Dropout(0.5))  # Add dropout layer for regularization
model.add(LSTM(64))  # Add another LSTM layer
model.add(Dropout(0.5))  # Add dropout layer for regularization
model.add(Dense(64, activation='relu'))  # Add a dense layer for better feature extraction
model.add(Dropout(0.5))  # Add dropout layer for regularization
model.add(Dense(1, activation='sigmoid'))

# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model on the training data
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2)
# Save the trained model
model.save('model.h5')

# Evaluating the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)

# predictions on the test data
y_pred_prob_1 = model.predict(X_test)
y_pred_1 = (y_pred_prob_1 > 0.5).astype(int)

print("Predicted values:", y_pred_1)
print("Actual values:", y_test)

# Evaluation Eetrics
report = classification_report(y_test, y_pred_1)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print("Classification Report:")
print(report)


#pickle file for model
import pickle
# Save the tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

pickle.dump(history, open("history.pkl", "wb"))

