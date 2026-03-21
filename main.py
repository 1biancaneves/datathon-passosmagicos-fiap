import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

st.set_page_config(page_title="Passos Mágicos - Dashboard", layout="wide")

@st.cache_data
def carregar_tudo():
    df = pd.read_csv('base_unificada.csv') if os.path.exists('base_unificada.csv') else pd.DataFrame()
    model = joblib.load('modelo_xgboost.pkl') if os.path.exists('modelo_xgboost.pkl') else None
    metrics = joblib.load('metrics.pkl') if os.path.exists('metrics.pkl') else None
    return df, model, metrics

df_total, xgb_model, metrics = carregar_tudo()

st.title("📊 Inteligência Preditiva - Passos Mágicos")
t1, t2, t3 = st.tabs(["Análise", "Modelo", "Diagnóstico Técnico"])

with t2:
    if metrics:
        st.header(f"⚙️ Performance com Limiar de {metrics['threshold']}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Acurácia", f"{metrics['acc']*100:.1f}%")
        c2.metric("Recall (Sensibilidade)", f"{metrics['rec']*100:.1f}%")
        c3.metric("Precision", f"{metrics['prec']*100:.1f}%")
        c4.metric("F1-Score", f"{metrics['f1']:.2f}")

        st.subheader("💡 O que mais impacta no risco?")
        feat_imp = pd.Series(xgb_model.feature_importances_, index=metrics['features']).sort_values(ascending=True)
        st.plotly_chart(px.bar(feat_imp, orientation='h', color_discrete_sequence=['#F7941E']))
    else: st.error("Modelo não encontrado.")

with t3:
    st.header("🧪 Validação Científica do Modelo")
    m1, m2, m3 = st.columns(3)
    m1.metric("AUC-ROC", f"{metrics['auc']:.2f}")
    m2.metric("KS (Separação)", f"{metrics['ks']:.2f}")
    m3.metric("Log Loss", f"{metrics['logloss']:.3f}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Curva de Aprendizado")
        if os.path.exists('learning_curve.png'): st.image('learning_curve.png')
    with col_b:
        st.subheader("SHAP Values (Explicabilidade)")
        if os.path.exists('shap_summary.png'): st.image('shap_summary.png')
