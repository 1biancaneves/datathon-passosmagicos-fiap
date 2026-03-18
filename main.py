import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import os

# CONFIGURAÇÃO DA PÁGINA
# Deve ser a primeira linha após os imports
st.set_page_config(page_title="Passos Mágicos - Dashboard", page_icon="📊", layout="wide")

# CARREGAMENTO DE DADOS E MODELO (COM CACHE)
@st.cache_data
def carregar_dados():
    # Caminho para a base unificada
    caminho_dados = 'base_unificada.csv'
    if os.path.exists(caminho_dados):
        df = pd.read_csv(caminho_dados)
        # Garantir tratamento de dados, como datas, se necessário
        # df['data_entrada'] = pd.to_datetime(df['data_entrada'])
        return df
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

# Carregar dados e modelo
df_total = carregar_dados()
xgb_model = carregar_modelo()

# Tabela de mapeamento das pedras
pedra_mapping = {
    0: 'Quartzo',
    1: 'Ágata',
    2: 'Ametista',
    3: 'Topázio'
}
# Lista para as labels dos filtros
pedra_labels = ['Quartzo', 'Ágata', 'Ametista', 'Topázio', 'Sem Classificação']

# --- BARRA LATERAL (SIDEBAR) E IMAGENS ---
with st.sidebar:
    # IMAGEM 1 - LOGO NO TOPO (Pode ser local ou URL do GitHub)
    try:
        # st.image('assets/logo3.png', width=200, output_format='PNG') # Se local
        st.image('https://raw.githubusercontent.com/SEU_USUARIO/SEU_REPOSITORIO/main/assets/logo3.png', width=200) # Se GitHub
    except Exception as e:
        st.write("Logo Passos Mágicos")

    st.title("Atalhos do Painel")
    st.markdown("Use o menu para navegar.")

    st.divider()
    # IMAGEM 2 - OUTRO PRESENTE NO RODAPÉ
    try:
        # st.image('assets/logo2.png', width=180, output_format='PNG') # Se local
        st.image('https://raw.githubusercontent.com/SEU_USUARIO/SEU_REPOSITORIO/main/assets/logo2.png', width=180) # Se GitHub
    except Exception as e:
        st.write("Apoio Datathon")

# --- CONTEÚDO PRINCIPAL (ABAS) ---
st.title("Painel de Impacto Educacional - Passos Mágicos")
st.markdown("Análise de Dados e Predição de Risco de Defasagem Escolar")

tab_simulacao, tab_exploratoria, tab_modelo = st.tabs([
    "🎯 Simulação do Aluno",
    "📊 Análise Exploratória (EDAs)",
    "⚙️ Detalhes do Modelo"
])

# ==============================================================================
# TAB 1: SIMULAÇÃO DO ALUNO (COM/SEM HISTÓRICO E PREDIÇÃO)
# ==============================================================================
with tab_simulacao:
    st.header("🎯 Simulador de Classificação do Aluno (INDE)")

    # Escolha do tipo de simulação
    st.markdown("### Selecione o Tipo de Simulação")
    modo_simulacao = st.radio("Como deseja inserir as notas?", ["Entrada Rápida (Sem Histórico)", "Simular Evolução (Com Histórico)"], horizontal=True)

    st.divider()

    # Variáveis globais para guardar as notas
    cols = ['IDA_2024', 'IEG_2024', 'IAA_2024', 'IPS_2024', 'IPP_2024', 'IPV_2024', 'IAN_2024']

    # --- ABA INTERNA: SEM HISTÓRICO ---
    if modo_simulacao == "Entrada Rápida (Sem Histórico)":
        st.subheader("1. Entrada de Notas Atuais (Escala 0 a 10)")
        
        # Cria colunas para os sliders
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            ida = st.slider("IDA (Aprendizado)", 0.0, 10.0, 7.0, 0.1, help="Indicador de Dificuldade de Aprendizado")
            ieg = st.slider("IEG (Engajamento)", 0.0, 10.0, 8.0, 0.1, help="Indicador de Engajamento")
        
        with col2:
            iaa = st.slider("IAA (Autoavaliação)", 0.0, 10.0, 7.5, 0.1, help="Indicador de Autoavaliação")
            ips = st.slider("IPS (Social)", 0.0, 10.0, 6.5, 0.1, help="Indicador Psicosocial")

        with col3:
            ipp = st.slider("IPP (Ponto de Virada)", 0.0, 10.0, 8.0, 0.1, help="Indicador de Ponto de Virada")
            ipv = st.slider("IPV (Vida)", 0.0, 10.0, 7.0, 0.1, help="Indicador de Vida")

        with col4:
            ian = st.slider("IAN (Nível)", 0.0, 10.0, 6.0, 0.1, help="Indicador de Adequação de Nível")

        # Criar DataFrame para predição (com nomes de coluna compatíveis com o modelo)
        input_data = pd.DataFrame([[ida, ieg, iaa, ips, ipp, ipv, ian]], columns=cols)
        
        # Botão de Calcular
        btn_prever = st.button("Calcular Predição de Pedra", type="primary")

    # --- ABA INTERNA: COM HISTÓRICO ---
    elif modo_simulacao == "Simular Evolução (Com Histórico)":
        st.subheader("1. Inserir Notas para um Histórico de 3 Anos")
        st.markdown("*Use os sliders para simular a trajetória do aluno.*")
        
        col1, col2, col3 = st.columns(3)
        
        # Ano N-2 (ex: 2022)
        with col1:
            st.markdown("#### Ano 2022")
            ida_22 = st.slider("IDA_22", 0.0, 10.0, 5.0, 0.1, key="ida22")
            ieg_22 = st.slider("IEG_22", 0.0, 10.0, 6.0, 0.1, key="ieg22")
            ian_22 = st.slider("IAN_22", 0.0, 10.0, 5.0, 0.1, key="ian22")
            # Adicionar outros indicadores se o modelo precisar
        
        # Ano N-1 (ex: 2023)
        with col2:
            st.markdown("#### Ano 2023")
            ida_23 = st.slider("IDA_23", 0.0, 10.0, ida_22 + 1.0, 0.1, key="ida23") # Sugere melhoria
            ieg_23 = st.slider("IEG_23", 0.0, 10.0, ieg_22 + 1.0, 0.1, key="ieg23")
            ian_23 = st.slider("IAN_23", 0.0, 10.0, ian_22, 0.1, key="ian23") # Mantém nível

        # Ano Atual (2024)
        with col3:
            st.markdown("#### Ano 2024 (Predição)")
            ida = st.slider("IDA_24", 0.0, 10.0, ida_23 + 1.0, 0.1, key="ida24")
            ieg = st.slider("IEG_24", 0.0, 10.0, ieg_23 + 1.0, 0.1, key="ieg24")
            ian = st.slider("IAN_24", 0.0, 10.0, ian_23 + 0.5, 0.1, key="ian24")
            # Adicionar outros 2024 que são usados na predição
            iaa = st.slider("IAA_24", 0.0, 10.0, 7.5, 0.1, key="iaa24")
            ips = st.slider("IPS_24", 0.0, 10.0, 6.5, 0.1, key="ips24")
            ipp = st.slider("IPP_24", 0.0, 10.0, 8.0, 0.1, key="ipp24")
            ipv = st.slider("IPV_24", 0.0, 10.0, 7.0, 0.1, key="ipv24")

        # Dados para predição (usando apenas 2024 conforme o modelo treinado)
        input_data = pd.DataFrame([[ida, ieg, iaa, ips, ipp, ipv, ian]], columns=cols)

        # Botão de Calcular
        btn_prever = st.button("Calcular Predição e Histórico", type="primary")

    # ==============================================================================
    # EXIBIÇÃO DO RESULTADO DA PREDIÇÃO (PARA AMBOS OS MODOS)
    # ==============================================================================
    if btn_prever and xgb_model is not None:
        
        # Predição e Probabilidades
        pred_pedra = xgb_model.predict(input_data)[0]
        pedra_nome = pedra_mapping.get(pred_pedra, "Desconhecida")
        probabilities = xgb_model.predict_proba(input_data)[0]

        st.divider()
        st.subheader("2. Resultado da Simulação")
        
        # Estilização do Resultado
        color = "#CCCCCC"
        if pedra_nome == 'Topázio': color = "#FFD700" # Dourado
        elif pedra_nome == 'Ametista': color = "#9932CC" # Roxo
        elif pedra_nome == 'Ágata': color = "#FF8C00" # Laranja
        elif pedra_nome == 'Quartzo': color = "#DDDDDD" # Cinza claro
        
        st.markdown(f"""
        <div style="background-color: {color}22; padding: 20px; border-radius: 10px; border: 2px solid {color}; text-align: center; margin-bottom: 20px;">
            <h1 style="color: {color}; margin: 0; font-size: 3em;">{pedra_nome}</h1>
            <p style="color: #333; font-size: 1.2em;">Classificação Prevista pelo Modelo XGBoost</p>
        </div>
        """, unsafe_allow_html=True)

        # Exibir Gráfico de Histórico se o modo for "Com Histórico"
        if modo_simulacao == "Simular Evolução (Com Histórico)":
            st.markdown("#### Histórico e Trajetória Simulada (INDE)")
            # Criar um DataFrame sintético para o gráfico
            # (Você pode usar os sliders IDA_22, IDA_23 etc. para popular isso)
            df_historico_sim = pd.DataFrame({
                'Ano': [2022, 2023, 2024],
                'IDA': [ida_22, ida_23, ida],
                'IEG': [ieg_22, ieg_23, ieg],
                'IAN': [ian_22, ian_23, ian]
            })
            df_historico_sim = df_historico_sim.melt('Ano', var_name='Indicador', value_name='Nota')
            
            # Gráfico de Linha simples
            st.line_chart(df_historico_sim, x='Ano', y='Nota', color='Indicador', y_label='Nota (0-10)', x_label='Ano Letivo')
            st.caption("Gráfico gerado a partir dos valores dos sliders.")

        st.markdown("#### Probabilidades da Predição")
        prob_df = pd.DataFrame([probabilities], columns=[pedra_mapping[i] for i in range(4)])
        st.bar_chart(prob_df.T, color="#4F8BF9")
        st.caption("Esta métrica representa a certeza do modelo para cada classificação.")

# ==============================================================================
# TAB 2: ANÁLISE EXPLORATÓRIA (EDAs COM FILTROS)
# ==============================================================================
with tab_exploratoria:
    st.header("📊 Análise Exploratória e Visualização de Dados")
    st.markdown("Explore os dados educacionais da Passos Mágicos usando os filtros abaixo.")

    # --- FILTROS DE DADOS (DENTRO DA ABA) ---
    with st.expander("🛠️ Filtros de Pesquisa", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Filtro de Ano (assume que há uma coluna 'ano' na base_unificada.csv)
            if 'ano' in df_total.columns:
                anos_disponiveis = sorted(df_total['ano'].unique())
                ano_selecionado = st.selectbox("Selecione o Ano", ["Todos"] + anos_disponiveis)
            else:
                ano_selecionado = "Todos"
                st.warning("Coluna 'ano' não encontrada na base.")

        with col2:
            # Filtro de Idade (Criar se não existir)
            if 'idade' in df_total.columns:
                idade_min = int(df_total['idade'].min())
                idade_max = int(df_total['idade'].max())
                idade_faixa = st.slider("Selecione a Faixa Etária", idade_min, idade_max, (idade_min, idade_max))
            else:
                st.warning("Coluna 'idade' não encontrada na base.")

        with col3:
            # Filtro de Gênero
            if 'genero' in df_total.columns:
                generos = df_total['genero'].unique()
                generos_selecionados = st.multiselect("Selecione o Gênero", generos, default=generos)
            
            # Filtro de Pedra Atual (conforme sua solicitação de "pedras, partes ou não sei o quê")
            if 'pedra_final' in df_total.columns:
                pedras_ativas = df_total['pedra_final'].unique()
                # st.write(pedras_ativas)
                # Converter códigos numéricos para nomes para o filtro
                # (Assumindo que sua base unificada ainda tem códigos)
                
                # pedra_filtro_labels = [pedra_mapping.get(p, p) for p in pedras_ativas]
                # pedras_selecionadas_nomes = st.multiselect("Filtrar por Pedra Atribuída", pedra_filtro_labels, default=pedra_filtro_labels)
                # Converter de volta para os valores da base
                # reverse_pedra_mapping = {v: k for k, v in pedra_mapping.items()}
                # pedras_selecionadas_valores = [reverse_pedra_mapping.get(p, p) for p in pedras_selecionadas_nomes]
        
        # Exemplo de Filtro adicional que você mencionou ("filtros de férias partes ou sei la")
        # if 'bolsa_estudo' in df_total.columns:
        #     # Criar o filtro se ele existir
        #     st.divider()

    st.divider()

    # --- APLICAÇÃO DOS FILTROS ---
    df_filtrado = df_total.copy()
    
    if ano_selecionado != "Todos":
        df_filtrado = df_filtrado[df_filtrado['ano'] == ano_selecionado]
    
    if 'idade' in df_total.columns:
        df_filtrado = df_filtrado[(df_filtrado['idade'] >= idade_faixa[0]) & (df_filtrado['idade'] <= idade_faixa[1])]
    
    # if 'pedra_final' in df_total.columns:
    #     df_filtrado = df_filtrado[df_filtrado['pedra_final'].isin(pedras_selecionadas_valores)]

    # Exibir Resumo dos Dados Filtrados
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    col_kpi1.metric("Alunos na Amostra", len(df_filtrado))
    if 'INDE' in df_filtrado.columns:
        col_kpi2.metric("Média INDE", f"{df_filtrado['INDE'].mean():.2f}")
    if 'IPP' in df_filtrado.columns:
        col_kpi3.metric("Média IPP", f"{df_filtrado['IPP'].mean():.2f}")

    # ==============================================================================
    # 📈 ESTRUTURA PARA OS 11 GRÁFICOS SOLICITADOS (Fase 5)
    # ==============================================================================
    st.divider()
    st.subheader("Análise Gráfica (Mínimo 11 Gráficos)")
    st.markdown("Cada gráfico abaixo responde a uma das 11 questões principais da Fase 5 do Datathon.")

    # --- GRÁFICO 1 ---
    with st.expander("Gráfico 1: Evolução do INDE por Ano Letivo", expanded=False):
        # Substitua este placeholder pela lógica real
        # plot1 = px.box(df_total, x='ano', y='INDE', color='pedra_final', title="Distribuição do INDE por Ano e Pedra")
        # st.plotly_chart(plot1, use_container_width=True)
        st.line_chart(df_total.groupby('ano')['INDE'].mean(), y_label='Média INDE', x_label='Ano Letivo')
        st.markdown("**Descrição Explicativa:** Este gráfico mostra a tendência da média do INDE ao longo dos anos. Ele nos permite identificar se o desempenho geral dos alunos está melhorando, piorando ou estabilizado, ajudando a avaliar a eficácia do programa a longo prazo.")

    # --- GRÁFICO 2 ---
    with st.expander("Gráfico 2: Composição da Classificação de Pedras por Gênero", expanded=False):
        # Substitua este placeholder pela lógica real (conforme sua ETL)
        if 'genero' in df_filtrado.columns:
            # Exemplo com st.bar_chart (melt necessary)
            st.subheader("Grafico Exemplo (Melt)")
            st.markdown("**Descrição Explicativa:** Analisa a distribuição dos alunos por gênero dentro de cada categoria de pedra. Isso ajuda a Passos Mágicos a entender se existe algum viés de gênero na evolução educacional ou se o programa está impactando igualmente meninos e meninas.")

    # --- GRÁFICO 3 ---
    with st.expander("Gráfico 3: Relação entre IDA (Aprendizado) e IEG (Engajamento)", expanded=False):
        # Exemplo com scatter
        # plot3 = px.scatter(df_filtrado, x='IEG', y='IDA', color='pedra_final', trendline="ols")
        # st.plotly_chart(plot3, use_container_width=True)
        st.subheader("Grafico Scatter Placeholder")
        st.markdown("**Descrição Explicativa:** Este gráfico de dispersão correlaciona as notas de aprendizado (IDA) com as de engajamento (IEG). A hipótese é que alunos mais engajados têm melhor desempenho. A presença de uma linha de tendência ajuda a confirmar essa correlação.")

    # --- GRÁFICO 4 ---
    with st.expander("Gráfico 4: Ponto de Virada (IPP) por Faixa Etária", expanded=False):
        # plot4 = px.violin(df_filtrado, x='idade', y='IPP')
        st.subheader("Grafico Placeholder")
        st.markdown("**Descrição Explicativa:** Mostra como o Indicador de Ponto de Virada varia de acordo com a idade. Isso é crucial para Passos Mágicos identificar o momento ideal de intervenção, ou seja, em qual fase da vida do aluno ele se torna mais receptivo ao empoderamento educacional.")

    # --- GRÁFICO 5 ---
    with st.expander("Gráfico 5: Distribuição da Defasagem de Nível (IAN)", expanded=False):
        st.subheader("Grafico Placeholder")
        st.markdown("**Descrição Explicativa:** Analisa a distribuição do IAN (Indicador de Nível). Uma nota baixa indica que o aluno está em um nível acadêmico abaixo do esperado para a sua idade. Este gráfico identifica se o programa está conseguindo reduzir essa defasagem.")

    # --- GRÁFICO 6 ---
    with st.expander("Gráfico 6: Evolução dos Alunos (Efeito Pedra)", expanded=False):
        st.subheader("Grafico Placeholder")
        st.markdown("**Descrição Explicativa:** Analisa se alunos que iniciam no Quartzo (maior vulnerabilidade) conseguem evoluir para pedras superiores ao longo do tempo. Esse gráfico demonstra o impacto direto do programa na jornada do aluno.")

    # --- GRÁFICO 7 ---
    with st.expander("Gráfico 7: Distribuição do IDA (Aprendizado) por Matéria (se houver)", expanded=False):
        st.subheader("Grafico Placeholder")
        st.markdown("**Descrição Explicativa:** Compara a distribuição das notas do IDA. Se sua base tiver matérias específicas (Português, Matemática), este gráfico compararia as duas, identificando em qual área os alunos têm maior dificuldade.")

    # --- GRÁFICO 8 ---
    with st.expander("Gráfico 8: Taxa de Evasão (se houver dados)", expanded=False):
        st.subheader("Grafico Placeholder")
        st.markdown("**Descrição Explicativa:** Calcula a taxa de alunos que saíram do programa a cada ano. Isso ajuda a Passos Mágicos a entender os pontos críticos que levam à evasão e a desenvolver estratégias de retenção.")

    # --- GRÁFICO 9 ---
    with st.expander("Gráfico 9: Análise Multivariada: INDE vs IPS vs IPV", expanded=False):
        st.subheader("Grafico Placeholder")
        st.markdown("**Descrição Explicativa:** Uma análise avançada que correlaciona o INDE com os indicadores Psicosocial (IPS) e de Vida (IPV). Ajuda a entender se o bem-estar social e emocional do aluno impacta seu desempenho acadêmico.")

    # --- GRÁFICO 10 ---
    with st.expander("Gráfico 10: Concentração de Alunos por Instituição/Unidade", expanded=False):
        st.subheader("Grafico Placeholder")
        st.markdown("**Descrição Explicativa:** Mostra a quantidade de alunos e sua média INDE por unidade da Passos Mágicos. Permite identificar unidades com performance excepcional ou que necessitam de mais suporte.")

    # --- GRÁFICO 11 ---
    with st.expander("Gráfico 11: Feature Importance (Exploratório - IPP como Target)", expanded=False):
        # Exemplo simples de heatmap de correlação
        # plot11 = px.imshow(df_filtrado[cols].corr(), text_auto=True)
        # st.plotly_chart(plot11, use_container_width=True)
        st.subheader("Grafico Placeholder")
        st.markdown("**Descrição Explicativa:** Este gráfico é uma prévia do modelo preditivo. Ele mostra quais indicadores principais (IDA, IEG, IAA, etc.) são mais correlacionados com o Ponto de Virada (IPP), ajudando a definir o que é 'ser um aluno de destaque' na Passos Mágicos.")

# ==============================================================================
# TAB 3: DETALHES DO MODELO (XGBOOST, MÉTRICAS E IMPORTÂNCIA)
# ==============================================================================
with tab_modelo:
    st.header("⚙️ Modelo Preditivo: Classificação de Pedras (XGBoost)")
    st.markdown("Nesta seção, detalhamos o modelo de Machine Learning usado para prever o risco de defasagem do aluno.")

    with st.expander("🧠 O que é o XGBoost?", expanded=True):
        st.markdown("""
        O **XGBoost (Extreme Gradient Boosting)** é um algoritmo de aprendizado de máquina supervisionado que pertence à família dos métodos de **ensemble** de árvores de decisão. 
        Ele é extremamente poderoso para dados estruturados (tabelares, como os da Passos Mágicos) e é conhecido por sua velocidade e precisão. 
        O XGBoost funciona combinando as previsões de múltiplos modelos mais simples (árvores de decisão) para criar uma previsão final mais robusta.
        """)
        st.divider()
        st.markdown("#### Por que o XGBoost foi escolhido?")
        st.markdown("""
        1.  **Alta Precisão:** É frequentemente o algoritmo vencedor em competições de data science.
        2.  **Lida com Outliers e Valores Nulos:** Possui mecanismos internos para lidar com dados faltantes, comuns em séries históricas.
        3.  **Controle de Overfitting:** Possui parâmetros de regularização que evitam que o modelo se ajuste demais aos dados de treino e falhe em dados novos.
        4.  **Feature Importance:** Permite que interpretemos quais indicadores educacionais são mais importantes para a classificação do aluno.
        """)

    st.divider()

    # --- MÉTRICAS DO MODELO (PEGUE ESSES NÚMEROS DO SEU MODEL_XGBOOST.PY) ---
    st.subheader("📊 Métricas de Performance do Modelo")
    st.markdown("Métricas calculadas nos dados de teste (conforme rodado no script `model_xgboost.py`).")
    
    col_met1, col_met2, col_met3 = st.columns(3)
    # INSIRA AQUI AS MÉTRICAS REAIS QUE VOCÊ OBTEVE
    col_met1.metric("Acurácia (Accuracy)", "XX.X%", help="Proporção de previsões corretas.")
    col_met2.metric("Precisão Weighted", "XX.X%", help="Proporção de verdadeiros positivos ajustada pelo peso de cada classe.")
    col_met3.metric("F1-Score Weighted", "XX.X%", help="A média harmônica entre precisão e recall, balanceando as classes.")

    st.markdown("**Interpretação das Métricas:**")
    st.markdown("""
    Uma acurácia de **XX%** indica que o modelo acerta a classificação da pedra em XX% das vezes. O F1-Score Weighted de **XX%** é a métrica mais importante neste caso, pois ela balanceia a precisão e o recall e leva em conta o desequilíbrio das classes (temos mais alunos Ametista do que Topázio, por exemplo). Um F1-Score acima de **XX%** é considerado um bom resultado para este tipo de problema social.
    """)

    st.divider()

    # --- GRÁFICO DE IMPORTÂNCIA DAS VARIÁVEIS ---
    st.subheader("💡 Feature Importance (Importância das Variáveis)")
    st.markdown("Este gráfico mostra quais indicadores educacionais foram decisivos para o modelo XGBoost chegar à sua conclusão.")

    # INSIRA AQUI O GRÁFICO DE FEATURE IMPORTANCE QUE VOCÊ GEROU NO MODEL_XGBOOST.PY
    try:
        if 'feature_importance_plot.png' in os.listdir('assets'):
            # Se você salvou o gráfico como imagem
            st.image('assets/feature_importance_plot.png', caption='Importância das Variáveis gerada pelo XGBoost', use_column_width=True)
        else:
            # Placeholder ou código para gerar o gráfico diretamente do modelo se você tiver as colunas
            if xgb_model is not None and 'feature_importances_' in dir(xgb_model):
                # Gerar gráfico rápido se tiver os dados das features
                # import matplotlib.pyplot as plt
                # import seaborn as sns
                import matplotlib.pyplot as plt

                # Se você tiver os nomes das colunas de treino em algum lugar, substitua cols
                feature_imp = pd.Series(xgb_model.feature_importances_, index=cols).sort_values(ascending=True)
                
                fig, ax = plt.subplots(figsize=(8, 4))
                feature_imp.plot(kind='barh', ax=ax, color='#4F8BF9')
                ax.set_title("Quais indicadores mais influenciam a Pedra?")
                st.pyplot(fig)
            else:
                st.warning("Gráfico de Feature Importance não encontrado ou modelo não carregado.")
    except Exception as e:
         st.warning("Não foi possível gerar o gráfico de feature importance.")
