import sys
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from data_processing.load_data import load_csv_data
from data_processing.process_data import process_data
from indexing.index_data import initialize_pinecone, create_index, index_data_in_batches
from models.sentence_transformer_model import load_model, get_embeddings, normalize_vector

load_dotenv()

def main():
    try:
        file_path = os.path.join(os.path.dirname(__file__), 'data', 'clean_climate_Fund_Database.csv')
        
        df = load_csv_data(file_path)
        if df is None:
            return
        
        logging.info("Colunas do DataFrame: %s", df.columns)
        logging.info("Primeiras linhas do DataFrame: %s", df.head())

        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df.dropna(inplace=True)
        
        data, metadata = process_data(df)
        if data is None or metadata is None:
            return
        
        logging.info("Exemplo de metadados: %s", metadata[0])

        pc = initialize_pinecone()
        if pc is None:
            return
        
        index_name = os.getenv('PINECONE_INDEX_NAME', 'fundos-verdes-index')
        dimension = 384
        index = create_index(pc, index_name, dimension)
        if index is None:
            return
        
        model = load_model()
        if model is None:
            return
        
        vectors = get_embeddings(model, data)
        if vectors is None:
            return
        
        normalized_vectors = [normalize_vector(vector) for vector in vectors]
        
        index_data_in_batches(index, normalized_vectors, metadata)
        
    except Exception as e:
        logging.error("Erro na execução do fluxo principal: %s", e)

if __name__ == "__main__":
    main()
