from sentence_transformers import SentenceTransformer
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def load_model():
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logging.info("Modelo carregado com sucesso!")
        return model
    except Exception as e:
        logging.error(f"Erro ao carregar o modelo: {e}")
        return None

def get_embeddings(model, texts):
    try:
        embeddings = model.encode(texts)
        logging.info("Embeddings gerados com sucesso!")
        return embeddings
    except Exception as e:
        logging.error(f"Erro ao gerar embeddings: {e}")
        return None

def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm

