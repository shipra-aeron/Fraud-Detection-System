from kafka import KafkaConsumer, KafkaProducer
import pandas as pd
import json
from utils.model_utils import load_model

model, scaler = load_model()

consumer = KafkaConsumer(
    'transaction_topic',
    bootstrap_servers='kafka:9092',
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='fraud-detection-group',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

producer = KafkaProducer(bootstrap_servers='kafka:9092', value_serializer=lambda v: json.dumps(v).encode('utf-8'))

def process_record(record):
    try:
        df = pd.DataFrame([record])
        features = df.drop(columns=['Class', 'Time'], errors='ignore')
        features_scaled = scaler.transform(features)
        predictions = model.predict(features_scaled)
        predictions_proba = model.predict_proba(features_scaled)[:, 1]

        response = {
            'predictions': predictions.tolist(),
            'probabilities': predictions_proba.tolist()
        }

        return response
    except Exception as e:
        return {'error': str(e)}

for message in consumer:
    record = message.value
    result = process_record(record)
    producer.send('predictions_topic', result)
