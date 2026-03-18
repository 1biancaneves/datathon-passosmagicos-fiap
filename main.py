import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="Passos Mágicos - Dashboard", page_icon="📊", layout="wide")

# CARREGAMENTO DE DADOS E MODELO (COM CACHE)
@st.cache_data
def carregar_dados():
    caminho_dados = 'base_unificada.csv'
    if os.path.exists(caminho_dados):
        return pd.read_csv(caminho_dados)
    else:
        st.error(f"Erro: Arquivo '{caminho_dados}' não encontrado.")
        return pd.DataFrame()

@st.cache_resource
def carregar_modelo():
    caminho_modelo = 'modelo_xgboost.pkl'
    if os.path.exists(caminho_modelo):
        with open(caminho_modelo, 'rb') as f:
            model = pickle.load(f)
        return model
    else:
        st.error(f"Erro: Arquivo '{caminho_modelo}' não encontrado.")
        return None

df_total = carregar_dados()
xgb_model = carregar_modelo()

# --- CABEÇALHO COM LOGOS (SEM SIDEBAR) ---
col_logo1, col_espaco, col_logo2 = st.columns([1, 4, 1])
with col_logo1:
    # Lembre de trocar para o link real do seu GitHub
    st.image('https://raw.githubusercontent.com/SEU_USUARIO/SEU_REPOSITORIO/main/assets/logo3.png', width=120)
with col_logo2:
    st.image('https://raw.githubusercontent.com/SEU_USUARIO/SEU_REPOSITORIO/main/assets/logo2.png', width=120)

st.title("Painel de Impacto Educacional - Passos Mágicos")
st.markdown("Análise de Dados e Predição de Risco de Defasagem Escolar")
st.divider()

# --- CONTEÚDO PRINCIPAL (ABAS) ---
tab_simulacao, tab_exploratoria, tab_modelo = st.tabs([
    "🎯 Simulação do Aluno",
    "📊 Análise Exploratória (EDAs)",
    "⚙️ Detalhes do Modelo"
])

# ==============================================================================
# TAB 1: SIMULAÇÃO DO ALUNO 
# ==============================================================================
with tab_simulacao:
    st.header("🎯 Simulador de Risco de Defasagem")

    modo_simulacao = st.radio("Como deseja inserir as notas?", ["Entrada Rápida (Sem Histórico)", "Simular Evolução (Com Histórico)"], horizontal=True)
    st.divider()

    # O MODELO EXIGE EXATAMENTE ESTES NOMES E NESTA ORDEM (sem o IAN)
    cols_modelo = ['IAA', 'IEG', 'IPS', 'IPP', 'IDA', 'IPV']

    if modo_simulacao == "Entrada Rápida (Sem Histórico)":
        st.subheader("1. Entrada de Notas Atuais (Escala 0 a 10)")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            iaa = st.slider("IAA (Autoavaliação)", 0.0, 10.0, 7.5, 0.1)
            ieg = st.slider("IEG (Engajamento)", 0.0, 10.0, 8.0, 0.1)
        with col2:
            ips = st.slider("IPS (Social)", 0.0, 10.0, 6.5, 0.1)
            ipp = st.slider("IPP (Ponto de Virada)", 0.0, 10.0, 8.0, 0.1)
        with col3:
            ida = st.slider("IDA (Aprendizado)", 0.0, 10.0, 7.0, 0.1)
            ipv = st.slider("IPV (Vida)", 0.0, 10.0, 7.0, 0.1)

        input_data = pd.DataFrame([[iaa, ieg, ips, ipp, ida, ipv]], columns=cols_modelo)
        btn_prever = st.button("Calcular Predição", type="primary")

    elif modo_simulacao == "Simular Evolução (Com Histórico)":
        st.subheader("1. Inserir Notas para Histórico")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Ano 2022")
            ida_22 = st.slider("IDA_22", 0.0, 10.0, 5.0, 0.1)
            ieg_22 = st.slider("IEG_22", 0.0, 10.0, 6.0, 0.1)
        with col2:
            st.markdown("#### Ano 2023")
            ida_23 = st.slider("IDA_23", 0.0, 10.0, ida_22 + 1.0, 0.1)
            ieg_23 = st.slider("IEG_23", 0.0, 10.0, ieg_22 + 1.0, 0.1)
        with col3:
            st.markdown("#### Ano 2024 (Predição)")
            ida = st.slider("IDA_24", 0.0, 10.0, ida_23 + 1.0, 0.1)
            ieg = st.slider("IEG_24", 0.0, 10.0, ieg_23 + 1.0, 0.1)
            # Para prever, o modelo precisa de todas as features de 2024
            iaa = st.slider("IAA_24", 0.0, 10.0, 7.5, 0.1)
            ips = st.slider("IPS_24", 0.0, 10.0, 6.5, 0.1)
            ipp = st.slider("IPP_24", 0.0, 10.0, 8.0, 0.1)
            ipv = st.slider("IPV_24", 0.0, 10.0, 7.0, 0.1)

        input_data = pd.DataFrame([[iaa, ieg, ips, ipp, ida, ipv]], columns=cols_modelo)
        btn_prever = st.button("Calcular Predição e Histórico", type="primary")

    # EXIBIÇÃO DO RESULTADO
    if btn_prever and xgb_model is not None:
        predicao = xgb_model.predict(input_data)[0] # 0 ou 1
        probabilidades = xgb_model.predict_proba(input_data)[0]

        st.divider()
        st.subheader("2. Resultado da Simulação")
        
        if predicao == 1:
            st.error("🚨 ALTO RISCO DE DEFASAGEM DETECTADO")
            st.markdown(f"**O modelo aponta uma probabilidade de {probabilidades[1]*100:.1f}% de defasagem para este aluno.** Recomenda-se intervenção.")
        else:
            st.success("✅ BAIXO RISCO DE DEFASAGEM")
            st.markdown(f"**O modelo aponta apenas {probabilidades[1]*100:.1f}% de chance de defasagem.** O aluno está em um bom caminho.")

        if modo_simulacao == "Simular Evolução (Com Histórico)":
            st.markdown("#### Histórico Simulado (Aprendizado vs Engajamento)")
            df_hist = pd.DataFrame({
                'Ano': [2022, 2023, 2024],
                'IDA': [ida_22, ida_23, ida],
                'IEG': [ieg_22, ieg_23, ieg]
            }).melt('Ano', var_name='Indicador', value_name='Nota')
            st.line_chart(df_hist, x='Ano', y='Nota', color='Indicador')

# ==============================================================================
# TAB 2: ANÁLISE EXPLORATÓRIA
# ==============================================================================
with tab_exploratoria:
    st.header("📊 Análise Exploratória e Visualização de Dados")
    
    with st.expander("🛠️ Filtros de Pesquisa", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            if 'Ano' in df_total.columns:
                anos_disponiveis = sorted(df_total['Ano'].unique())
                ano_selecionado = st.selectbox("Selecione o Ano", ["Todos"] + anos_disponiveis)
            else:
                ano_selecionado = "Todos"
        
        with col2:
            if 'Idade' in df_total.columns: # Ajuste o nome da coluna de idade se for diferente
                idade_min = int(df_total['Idade'].min())
                idade_max = int(df_total['Idade'].max())
                idade_faixa = st.slider("Faixa Etária", idade_min, idade_max, (idade_min, idade_max))

    df_filtrado = df_total.copy()
    if ano_selecionado != "Todos" and 'Ano' in df_filtrado.columns:
        df_filtrado = df_filtrado[df_filtrado['Ano'] == ano_selecionado]

    st.divider()
    st.subheader("Análise Gráfica (11 Gráficos)")
    
    with st.expander("Gráfico 1: Evolução do INDE por Ano Letivo", expanded=False):
        if 'Ano' in df_total.columns and 'INDE' in df_total.columns:
            st.line_chart(df_total.groupby('Ano')['INDE'].mean(), y_label='Média INDE')
        else:
            st.write("Colunas Ano ou INDE não encontradas para o Gráfico 1.")
            
    # Continue preenchendo os outros 10 expanders aqui...
    with st.expander("Gráfico 2 até 11: Insira seus gráficos aqui", expanded=False):
        st.write("Espaço reservado para os próximos gráficos.")

# ==============================================================================
# TAB 3: DETALHES DO MODELO
# ==============================================================================
with tab_modelo:
    st.header("⚙️ Modelo Preditivo: Risco de Defasagem (XGBoost)")
    st.markdown("O XGBoost foi escolhido por sua alta precisão com dados estruturados e lidar bem com valores ausentes.")
    
    col_met1, col_met2, col_met3 = st.columns(3)
    # INSIRA AS MÉTRICAS DO SEU MODEL_XGBOOST.PY AQUI
    col_met1.metric("Acurácia", "XX.X%")
    col_met2.metric("ROC-AUC", "XX.X%")
    col_met3.metric("F1-Score", "XX.X%")
    
    st.subheader("💡 Importância das Variáveis")
    if xgb_model is not None and hasattr(xgb_model, 'feature_importances_'):
        import matplotlib.pyplot as plt
        feature_imp = pd.Series(xgb_model.feature_importances_, index=cols_modelo).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        feature_imp.plot(kind='barh', ax=ax, color='#F7941E')
        ax.set_title("Quais indicadores mais pesam no risco de defasagem?")
        st.pyplot(fig)
    else:
        st.info("Rode seu arquivo model_xgboost.py e salve a imagem do feature importance na pasta assets.")

# --- RODAPÉ ---
st.divider()
st.markdown("<p style='text-align: center; color: gray;'>Desenvolvido para o Datathon Passos Mágicos - FIAP</p>", unsafe_allow_html=True)
