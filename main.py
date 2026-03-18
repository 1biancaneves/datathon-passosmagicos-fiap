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
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Dashboard Passos Mágicos - Fase 5", layout="wide", page_icon="🪄")

# --- ESTILO E CORES ---
# Cores oficiais sugeridas para o projeto
AZUL_PM = "#1D3D6F"
LARANJA_PM = "#F7941E"

st.markdown(f"""
    <style>
    .main {{ background-color: #f5f7f9; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 24px; }}
    .stTabs [data-baseweb="tab"] {{ height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; }}
    .stTabs [aria-selected="true"] {{ background-color: {AZUL_PM}; color: white; }}
    </style>
    """, unsafe_allow_html=True)

# --- CARREGAMENTO DE RECURSOS ---
@st.cache_data
def load_data():
    # Carrega a base unificada gerada no ETL
    if os.path.exists('base_unificada.csv'):
        df = pd.read_csv('base_unificada.csv')
        return df
    return pd.DataFrame()

df = load_data()

try:
    model = joblib.load('modelo_xgboost.pkl')
except:
    model = None

# --- CABEÇALHO ---
col_logo, col_titulo = st.columns([1, 5])
with col_logo:
    # Caminho para os assets no seu GitHub
    st.image("https://raw.githubusercontent.com/seu-usuario/seu-repositorio/main/assets/logo2.png", width=100)

with col_titulo:
    st.title("Associação Passos Mágicos - Análise de Impacto e Predição")
    st.write("Solução desenvolvida para o Datathon - Fase 5")

tab1, tab2, tab3 = st.tabs(["🎯 Simulação & Histórico", "📊 Análise Exploratória (11 Perguntas)", "🔬 Performance do Modelo"])

# ==============================================================================
# TAB 1: SIMULAÇÃO E HISTÓRICO
# ==============================================================================
with tab1:
    st.header("Simulador de Aluno e Trajetória Histórica")
    st.info("Preencha os indicadores dos últimos 3 anos para visualizar a evolução e o risco de defasagem para 2025.")
    
    # Dicionário de Indicadores
    with st.expander("📖 Glossário de Indicadores"):
        st.markdown("""
        * **IAA (Autoavaliação):** Atitude e interesse do aluno.
        * **IEG (Engajamento):** Entrega de tarefas e participação.
        * **IPS (Psicossocial):** Bem-estar emocional avaliado por psicólogos.
        * **IPP (Psicopedagógico):** Evolução cognitiva avaliada por pedagogos.
        * **IDA (Aprendizado):** Desempenho em Português, Matemática e Inglês.
        * **IPV (Ponto de Virada):** Maturidade e protagonismo do aluno.
        """)

    def col_inputs(ano):
        st.subheader(f"Dados de {ano}")
        iaa = st.number_input(f"IAA {ano}", 0.0, 10.0, 7.0, step=0.1, key=f"iaa_{ano}")
        ieg = st.number_input(f"IEG {ano}", 0.0, 10.0, 7.0, step=0.1, key=f"ieg_{ano}")
        ips = st.number_input(f"IPS {ano}", 0.0, 10.0, 7.0, step=0.1, key=f"ips_{ano}")
        ipp = st.number_input(f"IPP {ano}", 0.0, 10.0, 7.0, step=0.1, key=f"ipp_{ano}")
        ida = st.number_input(f"IDA {ano}", 0.0, 10.0, 7.0, step=0.1, key=f"ida_{ano}")
        ipv = st.number_input(f"IPV {ano}", 0.0, 10.0, 7.0, step=0.1, key=f"ipv_{ano}")
        return [iaa, ieg, ips, ipp, ida, ipv]

    c1, c2, c3 = st.columns(3)
    with c1: hist_22 = col_inputs(2022)
    with c2: hist_23 = col_inputs(2023)
    with c3: hist_24 = col_inputs(2024)

    if st.button("Gerar Diagnóstico Acadêmico", type="primary"):
        # Evolução Gráfica
        labels = ['IAA', 'IEG', 'IPS', 'IPP', 'IDA', 'IPV']
        fig_ev = go.Figure()
        fig_ev.add_trace(go.Scatter(x=labels, y=hist_22, name='2022', line=dict(dash='dot')))
        fig_ev.add_trace(go.Scatter(x=labels, y=hist_23, name='2023', line=dict(dash='dash')))
        fig_ev.add_trace(go.Scatter(x=labels, y=hist_24, name='2024', line=dict(width=4, color=AZUL_PM)))
        fig_ev.update_layout(title="Trajetória do Aluno (2022-2024)", yaxis_range=[0,10])
        st.plotly_chart(fig_ev, use_container_width=True)

        # Predição com base no ano mais recente (2024)
        if model:
            # Ordem das features conforme treinamento
            X_input = np.array([hist_24]) 
            prob = model.predict_proba(X_input)[0][1]
            risco = "ALTO" if prob > 0.5 else "BAIXO"
            
            st.subheader(f"Resultado da Predição: Risco {risco}")
            st.progress(prob)
            st.write(f"Probabilidade de defasagem futura: **{prob:.1%}**")
            if risco == "ALTO":
                st.warning("Recomendação: Intensificar acompanhamento Psicopedagógico (IPP) e reforço em IDA.")

# ==============================================================================
# TAB 2: ANÁLISE EXPLORATÓRIA (11 GRÁFICOS)
# ==============================================================================
with tab_eda:
    st.header("Painel de Indicadores - Fase 5")
    
    def plot_insight(fig, titulo, valor_pm):
        st.subheader(titulo)
        st.plotly_chart(fig, use_container_width=True)
        with st.container():
            st.markdown(f"**Valor Estratégico:** {valor_pm}")
        st.divider()

    if not df.empty:
        # 1. Distribuição INDE
        fig1 = px.histogram(df, x="INDE", color="Ano", barmode="overlay", title="1. Distribuição do INDE")
        plot_insight(fig1, "Variabilidade do INDE", "Identifica se a massa de alunos está evoluindo para notas maiores anualmente.")

        # 2. Alunos por Pedra
        fig2 = px.pie(df[df['Ano']==2024], names="Pedra", title="2. Classificação por Pedras (2024)")
        plot_insight(fig2, "Nível de Lapidação", "Exibe a maturidade atual do grupo; quanto mais Topázios, maior o sucesso do programa.")

        # 3. IDA por Gênero
        fig3 = px.box(df, x="Gênero", y="IDA", color="Gênero", title="3. Desempenho Acadêmico por Gênero")
        plot_insight(fig3, "Análise de Equidade", "Verifica se há necessidade de intervenções pedagógicas diferenciadas por gênero.")

        # 4. Escola Pública vs Privada
        fig4 = px.violin(df, x="Escola", y="INDE", box=True, title="4. Impacto do Tipo de Escola")
        plot_insight(fig4, "Influência Externa", "Avalia se o ambiente escolar externo (Público vs Privado) dita o sucesso dentro da PM.")

        # 5. IPV vs IDA
        fig5 = px.scatter(df, x="IDA", y="IPV", color="Pedra", trendline="ols", title="5. Correlação Ponto de Virada vs Aprendizado")
        plot_insight(fig5, "Gatilho de Mudança", "Valida se o 'Ponto de Virada' (IPV) realmente impulsiona as notas acadêmicas (IDA).")

        # 6. Engajamento por Fase
        fig6 = px.line(df.groupby('Fase')['IEG'].mean().reset_index(), x='Fase', y='IEG', markers=True, title="6. Engajamento Médio por Fase")
        plot_insight(fig6, "Risco de Abandono", "Detecta em quais fases o aluno perde o interesse, permitindo ações preventivas.")

        # 7. Idade vs Pedra
        fig7 = px.box(df, x="Pedra", y="Idade", title="7. Relação Idade e Evolução")
        plot_insight(fig7, "Adequação Etária", "Mostra se alunos mais velhos estão conseguindo atingir níveis avançados de lapidação.")

        # 8. Proporção de PV
        fig8 = px.bar(df.groupby('Ano')['Atingiu PV'].value_path().reset_index(), x='Ano', y='count', color='Atingiu PV', title="8. Evolução do Ponto de Virada")
        plot_insight(fig8, "Sucesso da Missão", "O objetivo final é o Ponto de Virada; este gráfico conta quantos 'venceram' por ano.")

        # 9. IPS por Pedra
        fig9 = px.strip(df, x="Pedra", y="IPS", color="Pedra", title="9. Estabilidade Psicossocial por Nível")
        plot_insight(fig9, "Base Emocional", "Prova que para subir de nível (Pedra), o aluno precisa de suporte emocional estável (IPS).")

        # 10. Evolução IAN
        fig10 = px.area(df.groupby(['Ano', 'IAN']).size().reset_index(name='Qtd'), x='Ano', y='Qtd', color='IAN', title="10. Adequação de Nível (IAN)")
        plot_insight(fig10, "Combate à Defasagem", "Monitora se estamos diminuindo a distância entre a fase ideal e a fase real do aluno.")

        # 11. Heatmap de Indicadores
        corr = df[['IAA','IEG','IPS','IPP','IDA','IPV']].corr()
        fig11 = px.imshow(corr, text_auto=True, title="11. Mapa de Influência entre Indicadores")
        plot_insight(fig11, "Insights de Gestão", "Revela quais indicadores 'puxam' os outros; ex: se IEG sobe, o IDA costuma subir também.")

# ==============================================================================
# TAB 3: MÉTRICAS DO MODELO
# ==============================================================================
with tab_model:
    st.header("Diagnóstico Técnico do Modelo XGBoost")
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Acurácia", "0.7054", help="Percentual total de acertos.")
        st.info("**Interpretação:** É um valor **Bom**. Em contextos sociais, prever comportamento com 70% de precisão é um resultado robusto.")
    with m2:
        st.metric("ROC-AUC", "0.7794", help="Capacidade de distinção entre as classes.")
        st.info("**Interpretação:** É um valor **Muito Bom**. O modelo consegue separar bem quem está em risco de quem não está.")
    with m3:
        st.metric("MCC", "0.3761", help="Coeficiente de Matthews.")
        st.info("**Interpretação:** É **Moderado**. Indica que o modelo é muito superior ao acaso, mesmo com classes desbalanceadas.")

    st.subheader("Importância das Variáveis (O que define o Risco?)")
    # Dados reais do seu treinamento
    feat_importances = pd.DataFrame({
        'Indicador': ['IDA', 'IPP', 'IEG', 'IPS', 'IAA', 'IPV'],
        'Importância': [14.28, 9.28, 5.96, 5.85, 4.96, 4.86]
    }).sort_values('Importância', ascending=True)
    
    fig_imp = px.bar(feat_importances, x="Importância", y="Indicador", orientation='h', color_discrete_sequence=[LARANJA_PM])
    st.plotly_chart(fig_imp)
    st.write("O **IDA (Aprendizado)** e o **IPP (Psicopedagógico)** são os maiores preditores. Isso significa que a defasagem é evitada principalmente com apoio escolar e pedagógico.")
