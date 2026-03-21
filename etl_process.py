import pandas as pd
import os

def clean_and_unify():
    file_name = 'BASE DE DADOS PEDE 2024 - DATATHON.xlsx'
    if not os.path.exists(file_name):
        print(f"Erro: Arquivo '{file_name}' não encontrado.")
        return

    print("Extraindo dados...")
    df22 = pd.read_excel(file_name, sheet_name='PEDE2022')
    df23 = pd.read_excel(file_name, sheet_name='PEDE2023')
    df24 = pd.read_excel(file_name, sheet_name='PEDE2024')

    def standardize(df, ano):
        cols_base = ['RA', 'IAA', 'IEG', 'IPS', 'IPP', 'IDA', 'IPV', 'IAN', 'INDE']
        mapping = {}
        for base in cols_base:
            for col in df.columns:
                c_name = str(col).upper()
                if base in c_name and "DESTAQUE" not in c_name:
                    if base == 'RA' and 'RA' != c_name[:2]: continue
                    if base not in mapping.values(): mapping[col] = base
        df_clean = df[list(mapping.keys())].copy().rename(columns=mapping)
        for b in cols_base:
            if b not in df_clean.columns: df_clean[b] = pd.NA
        df_clean['RA'] = df_clean['RA'].astype(str)
        df_clean = df_clean.dropna(subset=['RA'])
        df_clean['Ano'] = ano
        return df_clean

    print("Unificando e criando variáveis...")
    df_master = pd.concat([standardize(df22, 2022), standardize(df23, 2023), standardize(df24, 2024)], ignore_index=True)
    
    cols_numeric = ['IAA', 'IEG', 'IPS', 'IPP', 'IDA', 'IPV', 'IAN', 'INDE']
    for col in cols_numeric:
        df_master[col] = pd.to_numeric(df_master[col], errors='coerce')

    df_master = df_master.sort_values(['RA', 'Ano'])
    df_master['Risco_Defasagem'] = df_master['IAN'].apply(lambda x: 1 if x < 7 else 0)
    df_master['delta_IDA'] = df_master.groupby('RA')['IDA'].diff().fillna(0)
    df_master['is_new'] = (df_master.groupby('RA')['Ano'].rank(method='min') == 1).astype(int)

    df_master.to_csv('base_unificada.csv', index=False)
    print("base_unificada.csv gerada com sucesso.")

if __name__ == "__main__":
    clean_and_unify()
