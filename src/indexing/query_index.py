import os
import sys
import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import logging

logging.basicConfig(level=logging.INFO)

# Adicionar o caminho base do projeto ao sys.path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_dir = os.path.dirname(base_dir)
sys.path.extend([root_dir])

# Verificar se os caminhos estão corretos
logging.info("Base dir: %s", base_dir)
logging.info("Root dir: %s", root_dir)
logging.info("sys.path: %s", sys.path)

# Importações dos módulos personalizados
try:
    from models.sentence_transformer_model import load_model, get_embeddings
    from src.data_processing.load_data import load_csv_data
    from src.data_processing.process_data import process_data
except ImportError as e:
    logging.error("Erro ao importar os módulos: %s", e)
    sys.exit("Erro ao importar os módulos personalizados.")

load_dotenv()

def initialize_pinecone():
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY não encontrado. Verifique seu arquivo .env.")
    try:
        pc = Pinecone(api_key=pinecone_api_key)
        logging.info("Pinecone inicializado com sucesso!")
        return pc
    except Exception as e:
        logging.error("Erro ao inicializar o Pinecone: %s", e)
        return None

def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm

def query_index(index, query_vector, top_k=5):
    try:
        response = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
        logging.info("Consulta realizada com sucesso!")
        return response
    except Exception as e:
        logging.error("Erro ao consultar o índice: %s", e)
        return None

if __name__ == "__main__":
    pc = initialize_pinecone()
    if pc:
        index_name = os.getenv('PINECONE_INDEX_NAME', 'fundos-verdes-index')
        try:
            index = pc.Index(index_name)
        except Exception as e:
            logging.error("Erro ao acessar o índice Pinecone: %s", e)
            sys.exit("Erro ao acessar o índice Pinecone.")
        
        model = load_model()
        if model is None:
            sys.exit("Erro ao carregar o modelo.")
        
        query_text = "Sua consulta aqui"
        query_vector = get_embeddings(model, [query_text])
        if query_vector is None:
            sys.exit("Erro ao gerar embeddings para a consulta.")
        
        normalized_query_vector = normalize_vector(query_vector[0])
        normalized_query_vector = normalized_query_vector.tolist()  # Converter para lista
        
        response = query_index(index, normalized_query_vector)
        if response:
            logging.info("Resposta da consulta: %s", response)
