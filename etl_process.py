import pandas as pd
import numpy as np
import os

def clean_and_unify():
    # Usando caminhos relativos para funcionar em qualquer PC
    file_path = 'BASE DE DADOS PEDE 2024 - DATATHON.xlsx'
    if not os.path.exists(file_path):
        print(f"Erro: Arquivo '{file_path}' não encontrado.")
        return

    print(" Iniciando ETL: Unificando 2022, 2023 e 2024...")
    df22 = pd.read_excel(file_path, sheet_name='PEDE2022')
    df23 = pd.read_excel(file_path, sheet_name='PEDE2023')
    df24 = pd.read_excel(file_path, sheet_name='PEDE2024')

    def standardize(df, ano):
        cols_base = ['RA', 'IAA', 'IEG', 'IPS', 'IPP', 'IDA', 'IPV', 'IAN', 'INDE', 
                     'Fase', 'Pedra', 'Idade', 'Gênero', 'Escola', 'Atingiu PV']
        mapping = {}
        
        # Lógica para evitar colisão de nomes (Ex: IDA vs IDADE)
        df_cols = [str(c).strip().upper() for c in df.columns]
        
        for base in cols_base:
            base_up = base.upper()
            for i, col_name in enumerate(df.columns):
                c_up = df_cols[i]
                # Busca exata ou início da palavra (evita pegar IDA dentro de IDADE)
                if c_up == base_up or c_up.startswith(base_up + " "):
                    if "DESTAQUE" not in c_up:
                        mapping[col_name] = base
                        break
        
        df_clean = df[list(mapping.keys())].copy().rename(columns=mapping)
        for b in cols_base:
            if b not in df_clean.columns: df_clean[b] = np.nan
        
        # Limpeza de ruídos nos dados
        for col in ['Pedra', 'Fase', 'Escola']:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).replace(['INCLUIR', 'nan', 'NaN', 'None', '0'], np.nan)
        
        df_clean['RA'] = df_clean['RA'].astype(str)
        df_clean['Ano'] = ano
        return df_clean

    df_master = pd.concat([standardize(df22, 2022), standardize(df23, 2023), standardize(df24, 2024)], ignore_index=True)
    
    # Tratamento de categorias
    df_master['Tipo_Escola'] = df_master['Escola'].apply(lambda x: 'Pública' if 'PÚBLICA' in str(x).upper() or 'EE ' in str(x).upper() else 'Privada')
    df_master['Gênero'] = df_master['Gênero'].replace({'Menino': 'Masculino', 'Menina': 'Feminino'})
    
    # Garantir numéricos
    for c in ['IAA', 'IEG', 'IPS', 'IPP', 'IDA', 'IPV', 'IAN', 'INDE', 'Idade']:
        df_master[c] = pd.to_numeric(df_master[c], errors='coerce')

    # Engenharia de Risco
    df_master = df_master.sort_values(['RA', 'Ano'])
    df_master['Risco_Defasagem'] = df_master['IAN'].apply(lambda x: 1 if x < 7 else 0)
    df_master['delta_IDA'] = df_master.groupby('RA')['IDA'].diff().fillna(0)
    df_master['is_new'] = (df_master.groupby('RA')['Ano'].rank(method='min') == 1).astype(int)

    df_master.to_csv('base_unificada.csv', index=False)
    print("Sucesso: base_unificada.csv gerada com dados limpos!")

if __name__ == "__main__":
    clean_and_unify()
