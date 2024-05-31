import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

class PineconeUtil:
    @staticmethod
    def initialize_pinecone():
        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY não encontrado. Verifique seu arquivo .env.")
        try:
            pc = Pinecone(api_key=pinecone_api_key)
            print("Pinecone inicializado com sucesso!")
            return pc
        except Exception as e:
            print(f"Erro ao inicializar o Pinecone: {e}")
            return None
