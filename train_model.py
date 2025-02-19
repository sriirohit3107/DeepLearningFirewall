import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load CSIC 2010 Dataset
def load_csic2010_dataset(filepath="csic_database.csv"):
    # Read the CSV file without specifying a delimiter (default is comma)
    data = pd.read_csv(filepath, encoding='ISO-8859-1', header=None, names=['request'])
    # Apply simple heuristic to label malicious requests
    data['label'] = data['request'].apply(lambda x: 1 if ('SELECT' in x or 'UNION' in x or 'script' in x) else 0)
    return data

# Data Preprocessing
def preprocess_data(filepath):
    data = load_csic2010_dataset(filepath)
    X = data['request'].apply(lambda x: [ord(c) for c in str(x)[:100]])  # Convert HTTP requests to ASCII
    X = keras.preprocessing.sequence.pad_sequences(X, maxlen=100)  # Pad sequences
    y = LabelEncoder().fit_transform(data['label'])
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Load dataset
X_train, X_test, y_train, y_test = preprocess_data('csic_database.csv')

# Expand dimensions: CNN expects a 3D input (samples, 100, 1)
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

# Build CNN Model
model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(100, 1)),
    Dropout(0.2),
    Conv1D(64, kernel_size=3, activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Save Model
model.save('models/firewall_model.h5')
