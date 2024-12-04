import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def model_training():
    # Load your data
    df = pd.read_csv("dataset/creditcard_2023.csv")
    
    # Preprocess your data
    X = df.drop(columns=['Class', 'id'])
    y = df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train your model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate your model
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba)}")
    
    # Save your model and scaler
    joblib.dump(model, 'api/model.pkl')
    joblib.dump(scaler, 'api/scaler.pkl')

if __name__ == '__main__':
    model_training()
