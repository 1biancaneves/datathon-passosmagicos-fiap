import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

# CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="Passos Mágicos - Dashboard", page_icon="📊", layout="wide")

# CARREGAMENTO DE DADOS E MODELO
@st.cache_data
def carregar_dados():
    caminho_dados = 'base_unificada.csv'
    if os.path.exists(caminho_dados):
        df = pd.read_csv(caminho_dados)
        # Necessário para o modelo:
        df = df.sort_values(['RA', 'Ano'])
        df['delta_IDA'] = df.groupby('RA')['IDA'].diff().fillna(0)
        return df
    return pd.DataFrame()

@st.cache_resource
def carregar_modelo():
    if os.path.exists('modelo_xgboost.pkl'):
        return joblib.load('modelo_xgboost.pkl')
    return None

@st.cache_data
def carregar_metricas():
    if os.path.exists('metrics.pkl'):
        return joblib.load('metrics.pkl')
    return {"acc": 0.0, "auc": 0.0, "mcc": 0.0}

df_total = carregar_dados()
xgb_model = carregar_modelo()
metrics = carregar_metricas()

# --- INTERFACE ---
st.title("📊 Painel de Análise - Passos Mágicos")

tab_analise, tab_modelo = st.tabs(["Análise de Dados", "Modelo Preditivo"])

with tab_analise:
    st.header("Visão Geral dos Alunos")
    if not df_total.empty:
        st.dataframe(df_total.tail(10))
        # Adicione aqui seus gráficos de evolução (INDE, etc)
    else:
        st.warning("Aguardando base_unificada.csv")

with tab_modelo:
    st.header("⚙️ Modelo Preditivo: Risco de Defasagem")
    
    # Exibindo métricas reais capturadas no treino
    col1, col2, col3 = st.columns(3)
    col1.metric("Acurácia (Teste 2024)", f"{metrics['acc']*100:.1f}%")
    col2.metric("ROC-AUC", f"{metrics['auc']:.2f}")
    col3.metric("MCC", f"{metrics['mcc']:.2f}")

    st.subheader("💡 Importância dos Indicadores")
    if xgb_model is not None:
        features = ['IAA', 'IEG', 'IPS', 'IPP', 'IDA', 'IPV', 'delta_IDA']
        importances = xgb_model.feature_importances_
        feat_imp = pd.Series(importances, index=features).sort_values(ascending=True)
        
        fig_imp = px.bar(feat_imp, orientation='h', 
                         title="O que mais impacta no risco de atraso?",
                         labels={'value': 'Importância', 'index': 'Indicador'},
                         color_discrete_sequence=['#F7941E'])
        st.plotly_chart(fig_imp, use_container_width=True)
        
        # EXPLICAÇÃO TÉCNICA
        st.info("**Nota Técnica:** O modelo foi treinado com dados de 2022-2023 e validado com dados reais de 2024. "
                "A variável 'delta_IDA' mede a evolução do desempenho acadêmico, capturando se o aluno está melhorando ou piorando no tempo.")
