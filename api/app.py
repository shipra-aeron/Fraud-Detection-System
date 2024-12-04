from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

print("pandas1 version:", pd.__version__)

# Load the model and scaler
model = joblib.load('/app/model.pkl')
scaler = joblib.load('/app/scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Convert JSON data to DataFrame
        df = pd.DataFrame(data)

        # Drop 'Class' and 'id' columns if they exist
        features = df.drop(columns=['Class', 'id'])

        # Preprocess the data
        df_scaled = scaler.transform(features)

        # Make predictions
        predictions = model.predict(df_scaled)
        predictions_proba = model.predict_proba(df_scaled)[:, 1]

        # Prepare the response
        response = {
            'predictions': predictions.tolist(),
            'probabilities': predictions_proba.tolist()
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})
    
    
    
@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the 'file' part is present in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})  # Return an error if no file is uploaded
    
    # Get the file from the request
    file = request.files['file']  
    
    # Check if the filename is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'})  # Return an error if no file is selected
    
    # Check if the file is a CSV file
    if file and file.filename.endswith('.csv'):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file)  

        # Drop 'Class' and 'id' columns if they exist
        features = df.drop(columns=['Class', 'id'], errors='ignore')

        # Scale the features using the pre-fitted scaler
        features_scaled = scaler.transform(features)  

        # Make predictions
        predictions = model.predict(features_scaled) 
        predictions_proba = model.predict_proba(features_scaled)[:, 1]


        # Prepare the response containing the predictions and their probabilities
        response = {
            'predictions': predictions.tolist(),
            'probabilities': predictions_proba.tolist()
        }
        
        return jsonify(response)
    else:
        return jsonify({'error': 'Invalid file format. Please upload a CSV file.'})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
