import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier
from sklearn.metrics import *
from scipy.stats import ks_2samp

def train_model():
    df = pd.read_csv('base_unificada.csv').dropna(subset=['Risco_Defasagem', 'IAA', 'IDA'])
    features = ['IAA', 'IEG', 'IPS', 'IPP', 'IDA', 'IPV', 'delta_IDA', 'is_new']
    
    train = df[df['Ano'] < 2024]
    test = df[df['Ano'] == 2024]
    X_train, y_train = train[features], train['Risco_Defasagem']
    X_test, y_test = test[features], test['Risco_Defasagem']

    # Ajuste de peso para classes desbalanceadas
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)

    model = XGBClassifier(
        n_estimators=1000,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=ratio,
        eval_metric='aucpr',
        early_stopping_rounds=50,
        random_state=42
    )

    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)

    # PREVISÃO COM THRESHOLD DE 0.3 (Priorizando o Recall)
    y_proba = model.predict_proba(X_test)[:, 1]
    threshold = 0.3
    y_pred = (y_proba >= threshold).astype(int)

    # Cálculo de métricas solicitadas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    logloss = log_loss(y_test, y_proba)
    ks_stat, _ = ks_2samp(y_proba[y_test == 0], y_proba[y_test == 1])

    print(f"\n--- RELATÓRIO FINAL (Threshold: {threshold}) ---")
    print(f"Acurácia: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f} | AUC-ROC: {auc:.4f} | KS: {ks_stat:.4f}")
    print(f"Log Loss: {logloss:.4f} | Parou na iteração: {model.best_iteration}")

    # Gráficos
    results = model.evals_result()
    plt.figure()
    plt.plot(results['validation_0']['aucpr'], label='Treino')
    plt.plot(results['validation_1']['aucpr'], label='Teste 2024')
    plt.title('Curva de Aprendizado (PR-AUC)')
    plt.legend(); plt.savefig('learning_curve.png')

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    plt.figure(); shap.summary_plot(shap_values, X_test, show=False); plt.savefig('shap_summary.png')

    metrics = {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'auc': auc, 'ks': ks_stat, 
               'logloss': logloss, 'features': features, 'best_iter': model.best_iteration, 'threshold': threshold}
    joblib.dump(model, 'modelo_xgboost.pkl')
    joblib.dump(metrics, 'metrics.pkl')

if __name__ == "__main__":
    train_model()
