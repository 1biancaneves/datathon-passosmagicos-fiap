import streamlit as st
import pandas as pd
import joblib
import os
import exploratory_analysis as ea

st.set_page_config(page_title="Passos Mágicos - Dashboard", layout="wide")

# --- CARREGAMENTO ---
@st.cache_data
def load_data():
    return pd.read_csv('base_unificada.csv') if os.path.exists('base_unificada.csv') else pd.DataFrame()

df_total = load_data()
model = joblib.load('modelo_xgboost.pkl') if os.path.exists('modelo_xgboost.pkl') else None
metrics = joblib.load('metrics.pkl') if os.path.exists('metrics.pkl') else None

# --- FILTROS (Lógica: Se vazio = Tudo) ---
st.sidebar.header("Filtros")
if not df_total.empty:
    f_ano = st.sidebar.multiselect("Anos", options=sorted(df_total['Ano'].unique()))
    f_pedra = st.sidebar.multiselect("Pedra", options=df_total['Pedra'].dropna().unique())
    f_escola = st.sidebar.selectbox("Tipo de Escola", ["Todos", "Pública", "Privada"])

    dff = df_total.copy()
    if f_ano: dff = dff[dff['Ano'].isin(f_ano)]
    if f_pedra: dff = dff[dff['Pedra'].isin(f_pedra)]
    if f_escola != "Todos": dff = dff[dff['Tipo_Escola'] == f_escola]
else:
    dff = pd.DataFrame()

# --- ABAS ---
tab1, tab2, tab3 = st.tabs(["Análise de Impacto", "Simulador de Risco", "Detalhes Técnicos"])

with tab1:
    st.title("Impacto Social: A Jornada de Transformação")
    
    # Grid de 12 Gráficos
    funcs = [ea.get_evolucao_inde, ea.get_distribuicao_pedras, ea.get_performance_fase, 
             ea.get_radar_indicadores, ea.get_impacto_escola, ea.get_genero_dist,
             ea.get_idade_risco, ea.get_atingiu_pv, ea.get_heatmap, 
             ea.get_scatter_ieg_ida, ea.get_evolucao_ian, ea.get_top_superacao]
    
    for i in range(0, len(funcs), 2):
        c1, c2 = st.columns(2)
        with c1:
            f, t = funcs[i](dff)
            st.plotly_chart(f, use_container_width=True); st.info(t)
        if i+1 < len(funcs):
            with c2:
                f, t = funcs[i+1](dff)
                st.plotly_chart(f, use_container_width=True); st.info(t)

with tab2:
    st.header("Simulador de Alerta Antecipado")
    with st.form("simulador"):
        st.write("Insira os dados do aluno:")
        ca, cb, cc = st.columns(3)
        iaa = ca.number_input("IAA", 0.0, 10.0, 7.5, step=0.1)
        ieg = cb.number_input("IEG", 0.0, 10.0, 7.5, step=0.1)
        ips = cc.number_input("IPS", 0.0, 10.0, 7.5, step=0.1)
        
        cd, ce, cf = st.columns(3)
        ipp = cd.number_input("IPP", 0.0, 10.0, 7.5, step=0.1)
        ida = ce.number_input("IDA", 0.0, 10.0, 7.5, step=0.1)
        ipv = cf.number_input("IPV", 0.0, 10.0, 7.5, step=0.1)
        
        delta = st.slider("Evolução (Delta IDA)", -5.0, 5.0, 0.0)
        novo = st.checkbox("Aluno Novo?")
        
        if st.form_submit_button("Calcular Risco"):
            input_x = [[iaa, ieg, ips, ipp, ida, ipv, delta, 1 if novo else 0]]
            prob = model.predict_proba(input_x)[0][1]
            if prob >= 0.3:
                st.error(f"RISCO ALTO ({prob*100:.1f}%). Intervenção necessária.")
            else:
                st.success(f"RISCO BAIXO ({prob*100:.1f}%). Aluno estável.")

with tab3:
    st.header("Relatório Técnico")
    if metrics:
        st.write(f"**Recall (Proteção):** {metrics['rec']*100:.1f}%")
        st.info("""
        **Por que priorizamos o Recall?** Na Passos Mágicos, deixar um aluno em risco passar sem ajuda (Falso Negativo) 
        é um erro grave. Por isso calibramos o modelo para detectar quase todos os riscos (93%), mesmo que isso gere 
        alguns alarmes falsos. Isso prova o compromisso ético da solução.
        """)
        if os.path.exists('shap_summary.png'): st.image('shap_summary.png', caption="O que mais afeta o risco?")
