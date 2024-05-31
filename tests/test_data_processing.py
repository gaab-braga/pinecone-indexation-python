import pytest
import sys
import os

# Obter o caminho absoluto para o diretório raiz
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Adicionar o diretório raiz ao caminho de pesquisa de módulos
sys.path.append(root_dir)

# Agora você pode importar módulos de src
from src.data_processing.load_data import load_csv_data, process_data

def test_load_csv_data():
    df = load_csv_data('data/test_climate_Fund_Database.csv')
    assert df is not None
    assert 'Text' in df.columns

def test_process_data():
    df = load_csv_data('data/test_climate_Fund_Database.csv')
    data, metadata = process_data(df)
    assert data is not None
    assert metadata is not None
