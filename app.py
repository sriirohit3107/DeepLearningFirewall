from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('models/firewall_model.h5')

# Initialize Flask app
app = Flask(__name__ ,template_folder='app_py/templates')

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the HTTP request input from the form
    user_request = request.form['request']
    
    # Preprocess the input data (convert to ASCII and pad sequence)
    request_data = [ord(c) for c in str(user_request)[:100]]  # Convert to ASCII values
    padded_request = pad_sequences([request_data], maxlen=100)
    
    # Make prediction using the trained model
    prediction = model.predict(padded_request)
    result = "Malicious" if prediction[0] > 0.5 else "Normal"

    # Return the result to the template
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
