
                                                              Deep Learning Firewall

This project implements a deep learning-based firewall to classify HTTP requests as either Normal or Malicious. Using a Convolutional Neural Network (CNN) model trained on the CSIC 2010 dataset, the system learns to detect suspicious HTTP requests (e.g., those containing SQL injection patterns or script tags) and provides a web interface for real-time classification.

Project Structure

/project-root
├── app.py                      # Flask application (web interface for classification)
├── train_model.py              # Script to train the CNN model on CSIC 2010 data
├── csic_database.csv           # Dataset file (CSIC 2010 dataset)
├── /app_py
│   └── templates
│       └── index.html          # HTML template for the web interface
├── /models
│   └── firewall_model.h5       # Saved trained CNN model
├── requirements.txt            # List of Python dependencies
└── README.md                   # This file

Overview
Training:
train_model.py loads the CSIC 2010 dataset, preprocesses the HTTP request data (by converting the first 100 characters to ASCII values and padding the sequences), and trains a CNN model to classify requests. Malicious requests are identified based on a simple heuristic (if the request contains keywords such as SELECT, UNION, or script). The trained model is saved as models/firewall_model.h5.

Web Interface:
app.py is a Flask application that loads the saved model and provides an interface (via index.html) for users to input an HTTP request. The app preprocesses the input in the same way as the training data, uses the model to predict whether the request is "Normal" or "Malicious," and then displays the result.

Installation
Clone the Repository:

git clone https://github.com/sriirohit3107/DeepLearningFirewall.git
cd DeepLearningFirewall
Create a Virtual Environment (Optional but Recommended):

python -m venv venv

# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

Install Dependencies:

pip install -r requirements.txt
If you don’t have a requirements.txt file, you can install the following packages:

pip install numpy pandas tensorflow scikit-learn Flask
Training the Model
To train the CNN model:

Ensure the dataset file csic_database.csv is placed in the root directory.

Run the training script:
python train_model.py

This will train the model and save it as models/firewall_model.h5.
Running the Web Application
Once the model is trained, start the Flask web application:

python app.py
Then, open your web browser and navigate to http://127.0.0.1:5000/. Use the provided form to input an HTTP request and see whether it is classified as Normal or Malicious.

How It Works
Data Preprocessing:

Each HTTP request is converted to a sequence of ASCII values for the first 100 characters. The sequences are padded to a uniform length so they can be processed by the CNN.

Model Architecture:

The CNN model is built using two convolutional layers with dropout for regularization, followed by a dense layer and an output layer with sigmoid activation. The model is compiled with the Adam optimizer and binary crossentropy loss.

Prediction:

The Flask app (app.py) loads the trained model, preprocesses user input, and predicts the label:

If the predicted probability is above 0.5, the request is labeled as Malicious.
Otherwise, it is labeled as Normal.
Testing and Evaluation
The training script uses train_test_split to automatically split the dataset into training and testing sets (80% training, 20% testing). During training, the model’s performance on the test set is reported as val_accuracy and val_loss.

For additional evaluation, you can use:

y_pred = model.predict(X_test) > 0.5
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
This will provide precision, recall, and F1-score metrics for further analysis.

Future Enhancements : 

Dataset Expansion: Incorporate additional malicious HTTP request samples for more robust detection.
Advanced Feature Engineering: Use more sophisticated text processing techniques to extract additional features from HTTP requests.
Real-Time Monitoring: Extend the web interface to support real-time traffic monitoring and auto-blocking of malicious requests.
Model Improvements: Experiment with different neural network architectures (e.g., RNNs, Transformers) to improve classification accuracy.


Sample request : 

Malicous : 

![image](https://github.com/user-attachments/assets/ceaa0d94-d12d-4c74-81d3-b859b5cbcd1b)

Normal : 

![image](https://github.com/user-attachments/assets/2ea3a55b-71da-48a4-8d76-3641e379fd8d)
