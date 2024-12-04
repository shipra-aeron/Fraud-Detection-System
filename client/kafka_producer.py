import pandas as pd
from kafka import KafkaProducer
import json

def read_data(file_path):
    df = pd.read_csv(file_path)
    return df

def send_to_kafka(topic, data):
    producer = KafkaProducer(bootstrap_servers='kafka:9092', value_serializer=lambda v: json.dumps(v).encode('utf-8'))
    for record in data.to_dict(orient='records'):
        producer.send(topic, record)
    producer.flush()

if __name__ == "__main__":
    file_path = '/app/new_transactions.csv'
    topic = 'transaction_topic'
    data = read_data(file_path)
    send_to_kafka(topic, data)
