import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def process_data(df):
    try:
        text_column = df.select_dtypes(include='object').columns[0]
        data = df[text_column].tolist()
        metadata = df.drop(columns=[text_column]).to_dict(orient='records')
        logging.info("Dados de texto: %s", data[:5])
        logging.info("Exemplo de metadados: %s", metadata[:5])
        return data, metadata
    except Exception as e:
        logging.error("Erro ao processar os dados: %s", e)
        return None, None
