# Credit Card Fraud Detection System

This document provides detailed steps to run the Credit Card Fraud Detection System locally without using Docker.

## Prerequisites

- Python 3.9
- Virtual environment (venv)
- Apache Kafka and Zookeeper

## Directory Structure

```
```
Faud-detection-system/
├── api/
│   ├── app.py
│   ├── kafka_consumer.py
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── requirements.txt
│   ├── Dockerfile.api
├── client/
│   ├── kafka_producer.py
│   ├── client_app.py
│   ├── requirements.txt
│   ├── new_transactions.csv
│   ├── Dockerfile.client
├── frontend/
│   ├── index.html
│   ├── styles.css
│   ├── script.js
│   ├── Dockerfile.frontend
├── dataset/
│   ├── creditcard_2023.csv
├── README.md
├── docker-compose.yml
├── model_training.py
├── requirements.training.txt
├── nginx.conf

```

```

## Steps to Run the Application Locally
### Step 1: Set Up the Environment
1. Install Python and Virtual Environment:
Make sure you have Python installed on your machine. You can download it from python.org. Create a virtual environment for your project:
```
python -m venv venv
source venv/bin/activate [for mac]
venv\Scripts\activate [for windows]
```

2. Install Required Packages:
```
pip install -r api/requirements.txt
pip install -r client/requirements.txt
pip install -r api/requirements.training.txt
```


### Step 2: Install and Configure Kafka and Zookeeper
Download and install Kafka and Zookeeper from Apache Kafka. Follow the instructions to start both services:

1. Start Zookeeper
```
bin/zookeeper-server-start.sh config/zookeeper.properties
```

2. Start Kafka
```
bin/kafka-server-start.sh config/server.properties
```


### Step 3: Train the Model
Run the training script to create and save the model and scaler:
```
python api/model_training.py
```

This will generate model.pkl and scaler.pkl in the api directory.

### Step 4: Run the Flask API
Run the Flask app to start the API service:
```
python api/app.py
```

### Step 5: Run the Kafka Producer and Consumer
1. Start the Kafka Producer:
```
python client/kafka_producer.py
```

2. Start the Kafka Consumer.Open another terminal and run:
```
python api/kafka_consumer.py
```


### Step 6: Set Up the Frontend
Serve the frontend files using a simple HTTP server:
```
cd frontend
python -m http.server 8000
```


### Step 7: Run the Client App
Run the client app to process batch transactions and send them to the API for predictions:
```
python client/client_app.py
```

Access the Application
Frontend: Open your browser and go to http://localhost:8000 to access the frontend.
API: You can access the API endpoints at http://localhost:5000.

### Summary
1. Ensure Kafka and Zookeeper are running.
2. Train the model if needed.
3. Start the Flask API service.
4. Run the Kafka producer to send messages to the Kafka topic.
5. Run the Kafka consumer to process messages from the Kafka topic.
6. Serve the frontend files for user interaction.
7. Launch client_app.py to process batch transactions and send them to the API for predictions.