import plotly.express as px
import pandas as pd

def get_evolucao_inde(df):
    fig = px.line(df.groupby('Ano')['INDE'].mean().reset_index(), x='Ano', y='INDE', markers=True, title="1. Evolução do Sucesso (INDE)")
    return fig, "Este gráfico mostra se o desenvolvimento médio está subindo. O sucesso é uma linha ascendente."

def get_distribuicao_pedras(df):
    ordem = ['Quartzo', 'Ágata', 'Ametista', 'Topázio']
    fig = px.histogram(df.dropna(subset=['Pedra']), x='Ano', color='Pedra', barmode='group', category_orders={'Pedra': ordem}, title="2. Evolução das Classificações")
    return fig, "O objetivo é ver a transição: menos Quartzos e mais Topázios a cada ano."

def get_performance_fase(df):
    ordem = ['ALFA', 'FASE 1', 'FASE 2', 'FASE 3', 'FASE 4', 'FASE 5', 'FASE 6', 'FASE 7', 'FASE 8']
    fig = px.box(df, x='Fase', y='INDE', color='Ano', category_orders={'Fase': ordem}, title="3. Desempenho por Fase")
    return fig, "A altura do 'caixote' indica onde está a maioria das notas. Caixote alto = Fase saudável."

def get_radar_indicadores(df):
    inds = ['IAA', 'IEG', 'IPS', 'IPP', 'IDA', 'IPV']
    df_m = df.groupby('Ano')[inds].mean().reset_index().melt(id_vars='Ano')
    fig = px.line_polar(df_m, r='value', theta='variable', color='Ano', line_close=True, title="4. Perfil 360º do Aluno")
    return fig, "Mostra o equilíbrio entre Social e Acadêmico. Áreas mais largas indicam maior força."

def get_impacto_escola(df):
    fig = px.violin(df, x='Tipo_Escola', y='INDE', box=True, color='Tipo_Escola', title="5. Pública vs Privada")
    return fig, "Mede se a ONG está conseguindo igualar as oportunidades de desenvolvimento."

def get_genero_dist(df):
    fig = px.pie(df, names='Gênero', title="6. Equilíbrio de Gênero")
    return fig, "Distribuição demográfica dos alunos atendidos."

def get_idade_risco(df):
    fig = px.histogram(df, x='Idade', color='Risco_Defasagem', nbins=20, title="7. Idade Crítica de Risco")
    return fig, "Indica as faixas etárias que mais precisam de atenção psicossocial."

def get_atingiu_pv(df):
    fig = px.pie(df.dropna(subset=['Atingiu PV']), names='Atingiu PV', title="8. Alunos que Viraram o Jogo (PV)")
    return fig, "O Ponto de Virada indica protagonismo e autonomia do aluno."

def get_heatmap(df):
    cols = ['IAA', 'IEG', 'IPS', 'IPP', 'IDA', 'IPV', 'INDE']
    fig = px.imshow(df[cols].corr(), text_auto=".2f", title="9. Mapa de Influência")
    return fig, "Valores próximos de 1.0 mostram que os indicadores caminham juntos para o sucesso."

def get_scatter_ieg_ida(df):
    fig = px.scatter(df, x='IEG', y='IDA', color='Pedra', title="10. Engajamento vs Notas")
    return fig, "Alunos no topo direito são altamente esforçados e têm resultados acadêmicos."

def get_evolucao_ian(df):
    fig = px.area(df.groupby('Ano')['IAN'].mean().reset_index(), x='Ano', y='IAN', title="11. Necessidade de Apoio (IAN)")
    return fig, "O IAN mede a carência. Queremos ver o suporte aumentando com o tempo."

def get_top_superacao(df):
    top = df.sort_values('delta_IDA', ascending=False).head(10)
    fig = px.bar(top, x='RA', y='delta_IDA', color='delta_IDA', title="12. Top 10 Alunos Superação")
    return fig, "Destaque para os alunos que mais evoluíram suas notas no último ano."