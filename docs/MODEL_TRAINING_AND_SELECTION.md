# Model Training and Selection for Credit Card Fraud Detection

This document explains the thought process behind the model training and selection for the Credit Card Fraud Detection system.

## Objective

The objective of this project is to develop a machine learning model that can accurately detect fraudulent credit card transactions. The key challenges are handling the imbalanced dataset and ensuring the model is both precise and recall-effective.

## Data Understanding and Preprocessing

1. **Data Collection:**
   The dataset used for this project contains transactions made by credit cards in September 2013 by European cardholders. The dataset is highly imbalanced, with the positive class (frauds) accounting for 0.172% of all transactions.

2. **Data Preprocessing:**
   - **Handling Missing Values:** Check for any missing values and handle them appropriately. 
   - **Feature Scaling:** Since the dataset contains features (V1 to V28) resulting from a PCA transformation, they are already scaled. However, the 'Amount' feature needs to be scaled.
   - **Handling Imbalanced Data:** Use techniques such as SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.

## Model Selection

The following models were considered for training:

1. **Logistic Regression**:
   - **Pros**: Simple, interpretable, and performs well with linear relationships in data. Works decently with imbalanced datasets when combined with techniques like class weighting.
   - **Cons**: Struggles with capturing complex patterns and non-linear relationships in the data.

2. **Decision Tree**:
   - **Pros**: Intuitive and easy to interpret, can handle both numerical and categorical data without preprocessing.
   - **Cons**: Prone to overfitting, especially with small or imbalanced datasets, unless regularized (e.g., max depth, pruning).

3. **Random Forest**:
   - **Pros**: Reduces overfitting by aggregating multiple decision trees, robust to noise, and handles imbalanced data effectively when paired with sampling techniques like SMOTE.
   - **Cons**: Computationally heavier than single decision trees and less interpretable.

4. **Gradient Boosting**:
   - **Pros**: Achieves high accuracy by combining weak learners iteratively; handles class imbalance well and adapts to data distributions.
   - **Cons**: Slower to train compared to Random Forests and sensitive to hyperparameters.

5. **XGBoost (Extreme Gradient Boosting)**:
   - **Pros**: Highly efficient, supports missing values, handles large datasets well, and includes built-in mechanisms for class imbalance (e.g., scale_pos_weight).
   - **Cons**: Requires careful hyperparameter tuning; less interpretable without tools like SHAP or LIME.

6. **LightGBM (Light Gradient Boosting Machine)**:
   - **Pros**: Extremely fast and memory-efficient, handles large and imbalanced datasets effectively, and supports categorical features natively.
   - **Cons**: Sensitive to overfitting on small datasets if hyperparameters are not tuned properly.

7. **Support Vector Machine (SVM)**:
   - **Pros**: Effective for small to medium datasets and finds optimal hyperplanes for classification.
   - **Cons**: Computationally expensive for large datasets and sensitive to feature scaling.

8. **K-Nearest Neighbors (KNN)**:
   - **Pros**: Simple to understand and implement, makes no assumptions about data distribution.
   - **Cons**: Computationally expensive for large datasets, performance depends heavily on feature scaling and the choice of `k`.

9. **Neural Networks (Feedforward Neural Networks)**:
   - **Pros**: Capable of modeling complex relationships in data; effective for high-dimensional feature spaces.
   - **Cons**: Requires significant computational resources, hyperparameter tuning, and careful architecture design to avoid overfitting.

10. **Long Short-Term Memory (LSTM)**:
    - **Pros**: Specifically designed for sequential data; capable of capturing long-term dependencies in transaction history.
    - **Cons**: Not suitable for this dataset due to the absence of a `Time` column or sequential patterns.

11. **Graph Neural Networks (GNN)**:
    - **Pros**: Effective for graph-structured data, capturing relationships between entities (e.g., users, merchants, and transactions).
    - **Cons**: Relies on the availability of meaningful relational data, which is not inherently present in this dataset.

12. **CatBoost**:
    - **Pros**: Handles categorical features directly, robust to overfitting, and efficient for imbalanced datasets.
    - **Cons**: Computationally slower than LightGBM for very large datasets.


## Model Training and Evaluation

### Training Process

1. **Splitting the Data:**
   Split the dataset into training and testing sets. Use stratified splitting to ensure the training and testing sets have a similar class distribution.

   ```python
   from sklearn.model_selection import train_test_split

   X = df.drop(['Class', 'Time'], axis=1)
   y = df['Class']

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
   ```

2. **Feature Scaling:**

    Scale the features using standardization.

    ```python
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    ```
3. **Handling Imbalanced Data:**

    Apply SMOTE to the training data to balance the classes.

    ```python
    from imblearn.over_sampling import SMOTE

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    ```

4. **Model Training:**

    Train multiple models and evaluate their performance using cross-validation.

    ```python 
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import cross_val_score

    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(random_state=42)
    }

    results = []

    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=5, scoring='roc_auc')
        results.append({'Model': name, 'AUC-ROC': cv_scores.mean()})

    results_df = pd.DataFrame(results).sort_values(by='AUC-ROC', ascending=False)
    print(results_df)

    ```

5. **Model Evaluation:**
    Evaluate the performance of each model on the test set using metrics such as accuracy, precision, recall, F1 score, and AUC-ROC.

    ```python
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

    def evaluate_model(model, X_test, y_test):
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba)

        return accuracy, precision, recall, f1, auc_roc

    # Example evaluation for the best model
    best_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    best_model.fit(X_train_resampled, y_train_resampled)
    accuracy, precision, recall, f1, auc_roc = evaluate_model(best_model, X_test_scaled, y_test)

    print(f'Accuracy: {accuracy}')
    print(f'AUC-ROC: {auc_roc}')
    ```

6. **Final Model Selection:**
    Based on the evaluation metrics, select the model that provides the best trade-off between precision and recall, and has a high AUC-ROC score. In this case, the **Random Forest** classifier was selected due to its balance of performance, robustness to noise, and ability to handle class imbalance effectively. While models like Gradient Boosting and XGBoost offered comparable performance, Random Forest was chosen for its interpretability with tools like SHAP and its efficiency for the given dataset. Advanced techniques like GNN and LSTM were evaluated but deemed less suitable due to the lack of relational and temporal features in the dataset.

## Conclusion
By following this process, we ensure that the model selected for credit card fraud detection is robust, performs well on the imbalanced dataset, and generalizes well to new, unseen data. This careful approach to model training and selection is crucial for developing an effective fraud detection system.