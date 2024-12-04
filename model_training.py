import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import shap

def model_training():
    # Load dataset
    df = pd.read_csv("dataset/creditcard_2023.csv")
    
    # Preprocess data
    X = df.drop(columns=['Class', 'id'])
    y = df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply SMOTE for class balancing
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

    print("Original class distribution:", np.bincount(y))
    print("Resampled class distribution:", np.bincount(y_train_balanced))
        
    # Train Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba)}")

    # SHAP Integration
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled)

    # Visualize SHAP results
    shap.summary_plot(shap_values[1], X_test_scaled, feature_names=X.columns)
    shap.summary_plot(shap_values[1], X_test_scaled, feature_names=X.columns, plot_type="bar")
    
    # Save model and scaler
    joblib.dump(model, 'api/model.pkl')
    joblib.dump(scaler, 'api/scaler.pkl')

if __name__ == '__main__':
    model_training()
