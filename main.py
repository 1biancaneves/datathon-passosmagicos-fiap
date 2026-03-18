import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Passos Mágicos - Dashboard", layout="wide", page_icon="🪄")

# CSS para os fundos coloridos dos anos
st.markdown("""
    <style>
    .div-2024 { background-color: #ffffff; padding: 20px; border-radius: 10px; border: 1px solid #eee; margin-bottom: 10px; }
    .div-2023 { background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 10px; }
    .div-2022 { background-color: #e0e2e6; padding: 20px; border-radius: 10px; margin-bottom: 10px; }
    .div-atual { background-color: #e3f2fd; padding: 20px; border-radius: 10px; border: 2px solid #1d3d6f; margin-bottom: 10px; }
    .stNumberInput label { color: #1d3d6f !important; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CARREGAMENTO DE DADOS ---
@st.cache_data
def load_data():
    if os.path.exists('base_unificada.csv'):
        df = pd.read_csv('base_unificada.csv')
        # Garante que as colunas existam para evitar KeyError
        if 'Gênero' not in df.columns: df['Gênero'] = "Não Informado"
        if 'Pedra' not in df.columns: df['Pedra'] = "Não Informado"
        return df
    return pd.DataFrame()

df_total = load_data()

try:
    model = joblib.load('modelo_xgboost.pkl')
except:
    model = None

# --- 3. INTERFACE ---
st.title("📊 Monitoramento e Previsão - Passos Mágicos")

tab_sim, tab_eda, tab_model = st.tabs(["🎯 Simulador", "🔍 Análise Fase 5", "🔬 Métricas"])

# ==============================================================================
# ABA 1: SIMULAÇÃO COM CORES DIFERENCIADAS
# ==============================================================================
with tab_sim:
    st.header("Histórico do Aluno e Previsão de Risco")
    
    def criar_colunas_indicadores(ano, key_prefix):
        c1, c2, c3 = st.columns(3)
        iaa = c1.number_input(f"IAA ({ano})", 0.0, 10.0, 7.0, step=0.1, key=f"{key_prefix}_iaa")
        ieg = c2.number_input(f"IEG ({ano})", 0.0, 10.0, 7.0, step=0.1, key=f"{key_prefix}_ieg")
        ips = c3.number_input(f"IPS ({ano})", 0.0, 10.0, 7.0, step=0.1, key=f"{key_prefix}_ips")
        ipp = c1.number_input(f"IPP ({ano})", 0.0, 10.0, 7.0, step=0.1, key=f"{key_prefix}_ipp")
        ida = c2.number_input(f"IDA ({ano})", 0.0, 10.0, 7.0, step=0.1, key=f"{key_prefix}_ida")
        ipv = c3.number_input(f"IPV ({ano})", 0.0, 10.0, 7.0, step=0.1, key=f"{key_prefix}_ipv")
        return [iaa, ieg, ips, ipp, ida, ipv]

    # SEÇÃO 2024 (Branco)
    st.markdown('<div class="div-2024">', unsafe_allow_html=True)
    st.subheader("📅 Histórico 2024")
    n_2024 = criar_colunas_indicadores("2024", "y24")
    st.markdown
