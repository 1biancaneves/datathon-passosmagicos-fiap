import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, confusion_matrix

def train_model():
    # 1. CARREGAR DADOS
    df = pd.read_csv('base_unificada.csv')

    # 2. LIMPEZA 
    df = df.dropna(subset=['Risco_Defasagem', 'IAA', 'IDA'])

    # 3. FEATURE ENGINEERING (Evolução do aluno)
    df = df.sort_values(['RA', 'Ano'])
    df['delta_IDA'] = df.groupby('RA')['IDA'].diff().fillna(0)

    # 4. DEFINIÇÃO DAS FEATURES (7 indicadores agora)
    features = ['IAA', 'IEG', 'IPS', 'IPP', 'IDA', 'IPV', 'delta_IDA']

    # 5. SPLIT TEMPORAL (Evita olhar para o futuro)
    train = df[df['Ano'] < 2024]
    test = df[df['Ano'] == 2024]

    X_train = train[features]
    y_train = train['Risco_Defasagem']

    X_test = test[features]
    y_test = test['Risco_Defasagem']

    # 6. MODELO CONFIGURADO
    model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False
    )

    # 7. TREINAMENTO
    model.fit(X_train, y_train)

    # 8. AVALIAÇÃO REALISTA (Baseada em 2024)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    mcc = matthews_corrcoef(y_test, y_pred)

    print(f"--- Desempenho no Teste (Ano 2024) ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"MCC: {mcc:.4f}")
    
    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))

    # 9. SALVAR MODELO E MÉTRICAS
    joblib.dump(model, 'modelo_xgboost.pkl')
    # Salvamos as métricas para o Streamlit ler depois
    metrics = {'acc': acc, 'auc': auc, 'mcc': mcc}
    joblib.dump(metrics, 'metrics.pkl')
    
    print("\nModelo e métricas salvos com sucesso!")

if __name__ == "__main__":
    train_model()
