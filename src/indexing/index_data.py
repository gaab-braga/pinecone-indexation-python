import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import itertools
import logging
from src.database.database import read_data_from_db

load_dotenv()
logging.basicConfig(level=logging.INFO)

def initialize_pinecone():
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY não encontrado. Verifique seu arquivo .env.")
    
    try:
        pc = Pinecone(api_key=pinecone_api_key)
        logging.info("Pinecone inicializado com sucesso!")
        return pc
    except Exception as e:
        logging.error(f"Erro ao inicializar o Pinecone: {e}")
        return None

def create_index(pc, index_name, dimension, metric='cosine'):
    try:
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            logging.info(f"Índice {index_name} criado com sucesso!")
        else:
            logging.info(f"Índice {index_name} já existe.")
        
        return pc.Index(index_name)
    except Exception as e:
        logging.error(f"Erro ao criar ou acessar o índice: {e}")
        return None

def chunks(iterable, batch_size=100):
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))

def validate_metadata(metadata):
    if not isinstance(metadata, dict):
        logging.warning(f"Metadados inválidos: {metadata}. Deve ser um dicionário.")
        return {}
    return metadata

def index_data_in_batches(index, vectors, metadata_list, batch_size=100):
    try:
        for i, (batch, metadata_batch) in enumerate(zip(chunks(vectors, batch_size), chunks(metadata_list, batch_size))):
            to_upsert = [
                {
                    "id": str(i * batch_size + j),
                    "values": vector.tolist(),  # Certificar que os vetores estão na forma de lista
                    "metadata": validate_metadata(metadata)
                }
                for j, (vector, metadata) in enumerate(zip(batch, metadata_batch))
            ]
            index.upsert(vectors=to_upsert)
        logging.info("Dados indexados com sucesso em lotes!")
    except Exception as e:
        logging.error(f"Erro ao indexar dados: {e}")

if __name__ == "__main__":
    pc = initialize_pinecone()
    if pc:
        index_name = "seu_index_name"
        vectors, metadata_list, dimension = read_data_from_db()
        index = create_index(pc, index_name, dimension)
        if index:
            index_data_in_batches(index, vectors, metadata_list)
