import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def organize_csv(file_path):
    try:
        df = pd.read_csv(file_path, header=0, skipinitialspace=True, on_bad_lines='skip')
        logging.info("Primeiras linhas do DataFrame antes de organizar:\n%s", df.head())
        
        df.dropna(axis=1, how='all', inplace=True)
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        
        logging.info("Primeiras linhas do DataFrame após organizar:\n%s", df.head())
        return df
    except Exception as e:
        logging.error("Erro ao organizar o CSV: %s", e)
        return None

if __name__ == "__main__":
    file_path = '../data/climate_Fund_Database.csv'
    df = organize_csv(file_path)
    if df is not None:
        df.to_csv('../data/clean_climate_Fund_Database.csv', index=False)
        logging.info("CSV organizado salvo com sucesso!")
