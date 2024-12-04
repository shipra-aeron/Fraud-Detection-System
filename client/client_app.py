import requests
import pandas as pd
import json

print("pandas2 version:", pd.__version__)

# Constants
BATCH_SIZE = 100  # Define the batch size

# Read the CSV file
df = pd.read_csv('/app/new_transactions.csv')

# Define the API endpoint
url = 'http://fraud_detection_api:5000/predict'

# Initialize an empty list to store responses
all_responses = []

# Process data in batches
for start in range(0, len(df), BATCH_SIZE):
    end = start + BATCH_SIZE
    batch = df.iloc[start:end]

    # Convert DataFrame batch to JSON
    data = batch.to_dict(orient='records')

    # Make the request
    response = requests.post(url, json=data)

    # Check if the request was successful
    if response.status_code == 200:
        # Append the response to the list
        all_responses.extend(response.json().get('predictions', []))
    else:
        print(f"Batch {start} to {end} failed with status code: {response.status_code}")

# Write the combined response to a file
with open('/app/response.json', 'w') as file:
    json.dump(all_responses, file, indent=4)

print("Response written to /app/response.json")
