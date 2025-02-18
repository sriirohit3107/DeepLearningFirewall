import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
model = tf.keras.models.load_model('models/firewall_model.h5')

def is_malicious_request(http_request):
    # Convert the request to a numeric representation
    request_array = np.array([ord(c) for c in http_request[:100]])
    padded_request = pad_sequences([request_array], maxlen=100)
    
    # Make prediction
    prediction = model.predict(padded_request)
    return bool(prediction > 0.5)
