import os
import pandas as pd

def load_csv_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"O arquivo {file_path} não foi encontrado.")
    
    try:
        df = pd.read_csv(file_path, delimiter=',', header=0, skipinitialspace=True, on_bad_lines='skip')
        print("CSV carregado com sucesso!")
        return df
    except Exception as e:
        print(f"Erro ao carregar o CSV: {e}")
        return None

def process_data(df):
    try:
        text_column = df.select_dtypes(include='object').columns[0]
        data = df[text_column].tolist()
        metadata = df.drop(columns=[text_column]).to_dict(orient='records')
        return data, metadata
    except Exception as e:
        print(f"Erro ao processar os dados: {e}")
        return None, None