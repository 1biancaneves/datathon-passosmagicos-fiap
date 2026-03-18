import pandas as pd

def clean_and_unify():
    # Caminho do seu arquivo (ajustado para r'' para evitar erro de barras no Windows)
    file_path = r'C:\Users\Cliente\Downloads\datathon-passosmagicos-fiap\BASE DE DADOS PEDE 2024 - DATATHON.xlsx'
    
    # Lendo cada aba
    df22 = pd.read_excel(file_path, sheet_name='PEDE2022')
    df23 = pd.read_excel(file_path, sheet_name='PEDE2023')
    df24 = pd.read_excel(file_path, sheet_name='PEDE2024')

    def standardize(df, ano):
        # Indicadores que queremos extrair
        cols_base = ['RA', 'IAA', 'IEG', 'IPS', 'IPP', 'IDA', 'IPV', 'IAN', 'INDE']
        mapping = {}
        
        # Lógica rigorosa: para cada indicador base, procuramos a melhor coluna no Excel
        for base in cols_base:
            for col in df.columns:
                c_name = str(col).upper()
                # Regra: O nome base tem que estar na coluna, mas NÃO pode ser coluna de "Destaque"
                if base in c_name and "DESTAQUE" not in c_name:
                    # Caso especial para o RA não pegar outras colunas acidentalmente
                    if base == 'RA' and 'RA' != c_name[:2]:
                        continue
                    
                    mapping[col] = base
                    break # Encontrou a coluna principal, pula para o próximo indicador base

        # Filtra apenas as colunas encontradas e renomeia
        df_clean = df[list(mapping.keys())].copy()
        df_clean = df_clean.rename(columns=mapping)
        
        # Garante que as colunas base existam (se faltar alguma em algum ano, preenche com NaN)
        for b in cols_base:
            if b not in df_clean.columns:
                df_clean[b] = pd.NA

        # Limpeza básica
        df_clean['RA'] = df_clean['RA'].astype(str)
        df_clean = df_clean.dropna(subset=['RA'])
        df_clean['Ano'] = ano
        return df_clean

    # Agora o concat não deve dar erro pois as colunas são únicas
    df_master = pd.concat([standardize(df22, 2022), 
                           standardize(df23, 2023), 
                           standardize(df24, 2024)], ignore_index=True)

    # Converter colunas para numérico
    cols_numeric = ['IAA', 'IEG', 'IPS', 'IPP', 'IDA', 'IPV', 'IAN', 'INDE']
    for col in cols_numeric:
        df_master[col] = pd.to_numeric(df_master[col], errors='coerce')

    # Criar o Target (Risco de Defasagem)
    # Usamos o IAN: se for menor que 7 (ou seja, se houver defasagem moderada ou severa)
    df_master['Risco_Defasagem'] = df_master['IAN'].apply(lambda x: 1 if (pd.notnull(x) and x < 7) else 0)
    
    # Salvar
    df_master.to_csv('C:\\Users\\Cliente\\Downloads\\datathon-passosmagicos-fiap\\base_unificada.csv', index=False)
    print("Sucesso! Arquivo 'base_unificada.csv' gerado sem erros de índice.")

if __name__ == "__main__":
    clean_and_unify()