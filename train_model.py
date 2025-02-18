import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load Dataset from CSV (adjust file path as needed)
def load_dataset(filepath="your_data.csv"):
    # Read the CSV file
    data = pd.read_csv(filepath, encoding='ISO-8859-1')
    
    # Extract request and classification (assuming last column is classification)
    data = data[['Method', 'User-Agent', 'Accept', 'Content-Type', 'classification', 'URL']]
    
    # Convert HTTP requests to a list of ASCII values (limited to first 100 characters)
    data['request'] = data.apply(lambda x: ' '.join([str(x[col]) for col in ['Method', 'User-Agent', 'Accept', 'Content-Type', 'URL']]), axis=1)
    data['request'] = data['request'].apply(lambda x: [ord(c) for c in str(x)[:100]])  # Convert to ASCII values (only first 100 characters)
    
    # Assign labels: 0 for normal, 1 for malicious
    data['label'] = data['classification'].apply(lambda x: 1 if x == 'malicious' else 0)
    
    return data

# Data Preprocessing
def preprocess_data(filepath):
    data = load_dataset(filepath)
    
    # Padding sequences to have same length
    X = pad_sequences(data['request'], maxlen=100)  # Adjust maxlen based on your input length needs
    
    # Label encoding for classification labels
    y = LabelEncoder().fit_transform(data['label'])
    
    # Split into training and testing data
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Load dataset
X_train, X_test, y_train, y_test = preprocess_data('your_data.csv')

# Build CNN Model
model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(100, 1)),
    Dropout(0.2),
    Conv1D(64, kernel_size=3, activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Reshape data for CNN (input shape: [samples, time_steps, features])
X_train = np.expand_dims(X_train, -1)  # Reshape to (samples, time_steps, 1)
X_test = np.expand_dims(X_test, -1)  # Reshape to (samples, time_steps, 1)

# Train Model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Save Model
model.save('models/firewall_model.h5')
