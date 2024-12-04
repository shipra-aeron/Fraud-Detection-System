
# Running the Application with Docker - Credit Card Fraud Detection System

This document provides detailed steps to run the Credit Card Fraud Detection System using Docker.

## Prerequisites

- Docker
- Docker Compose

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

## Steps to Run the Application with Docker
### Step 1: Build and Run the Docker Containers
Build and run the Docker containers:

```
docker-compose pull
docker-compose up --build
```

This command will build and start all the services defined in the docker-compose.yml file, including Kafka, Zookeeper, the API service, the client service, and the frontend service.

### Step 2: Access the Application
Frontend: Open your browser and go to http://localhost:8000 to access the frontend.
1. You can access the API endpoints at http://localhost:5000.
2. Docker Configuration

docker-compose.yml:
```
version: '3.9'

networks:
  fraud_detection_network:
    driver: bridge

services:
  zookeeper:
    image: bitnami/zookeeper:latest
    environment:
      - ZOO_ENABLE_AUTH=yes
      - ZOO_SERVER_USERS=user
      - ZOO_SERVER_PASSWORDS=password
    ports:
      - "2181:2181"
    networks:
      - fraud_detection_network

  kafka:
    image: bitnami/kafka:latest
    environment:
      - KAFKA_BROKER_ID=1
      - KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181
      - ALLOW_PLAINTEXT_LISTENER=yes
      - KAFKA_LISTENERS=PLAINTEXT://:9092
      - KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://kafka:9092
    ports:
      - "9092:9092"
    depends_on:
      - zookeeper
    networks:
      - fraud_detection_network

  api:
    build:
      context: ./api
      dockerfile: Dockerfile.api
    ports:
      - "5000:5000"
    networks:
      - fraud_detection_network
    volumes:
      - ./api:/app

  client:
    build:
      context: ./client
      dockerfile: Dockerfile.client
    depends_on:
      - api
    networks:
      - fraud_detection_network
    volumes:
      - ./client:/app

  frontend:
    image: nginx:latest
    volumes:
      - ./frontend:/usr/share/nginx/html
      - ./nginx.conf:/etc/nginx/conf.d/default.conf
    ports:
      - "8080:80"
    networks:
      - fraud_detection_network

```

## Summary
The docker-compose up --build command will build and start all services.

Access the frontend at http://localhost:8080.

Access the API at http://localhost:5000.

By following these steps, you can run the Credit Card Fraud Detection System using Docker containers. This setup ensures all components work together to provide the full functionality of the application.