import sqlite3
import numpy as np

def read_data_from_db(db_path):
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    
    cursor.execute("SELECT id, vector, genre FROM sua_tabela")
    rows = cursor.fetchall()
    
    vectors = []
    metadata_list = []
    
    max_dimension = 0
    for row in rows:
        vector = np.fromstring(row[1], sep=',').tolist()
        if len(vector) > max_dimension:
            max_dimension = len(vector)
        metadata = {"genre": row[2]}
        vectors.append(vector)
        metadata_list.append(metadata)
    
    connection.close()
    return vectors, metadata_list, max_dimension
