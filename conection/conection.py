import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

def connect_to_pinecone():
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY não encontrado. Verifique seu arquivo .env.")
    
    try:
        pc = Pinecone(api_key=pinecone_api_key)
        print("Conectado ao Pinecone com sucesso!")
        return pc
    except Exception as e:
        print(f"Erro ao conectar ao Pinecone: {e}")
        return None

if __name__ == "__main__":
    connect_to_pinecone()
