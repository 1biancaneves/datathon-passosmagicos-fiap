import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

def train_model():
    df = pd.read_csv('C:\\Users\\Cliente\\Downloads\\datathon-passosmagicos-fiap\\base_unificada.csv').dropna(subset=['Risco_Defasagem', 'IAA', 'IDA'])
    
    # Features baseadas nos indicadores da Passos Mágicos
    features = ['IAA', 'IEG', 'IPS', 'IPP', 'IDA', 'IPV']
    X = df[features]
    y = df['Risco_Defasagem']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # XGBoost com foco em evitar Overfitting e métricas de ranking
    model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False
    )

    model.fit(X_train, y_train, 
              eval_set=[(X_test, y_test)], 
              verbose=False)

    # Avaliação
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"MCC: {matthews_corrcoef(y_test, y_pred):.4f}")
    
    # Matriz de Confusão
    print("\\nMatriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))

    # Feature Importance (Gain / Information)
    importances = model.get_booster().get_score(importance_type='gain')
    print("\nImportância das Variáveis (Gain):", importances)

    # Salva o modelo para o main.py
    joblib.dump(model, 'C:\\Users\\Cliente\\Downloads\\datathon-passosmagicos-fiap\\modelo_xgboost.pkl')
    print("\nModelo guardado como 'modelo_xgboost.pkl'")

if __name__ == "__main__":
    train_model()