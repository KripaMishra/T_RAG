# data ingestion-----------------------------------------------
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, connections, utility
from sentence_transformers import SentenceTransformer
from tree_indexer import TreeIndex
from data_prep import *

import numpy as np

# Connect to Milvus
connections.connect(host='localhost', port='19530')

# Define collection schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="chunk_id", dtype=DataType.INT64),
    FieldSchema(name="section_id", dtype=DataType.INT64),
    FieldSchema(name="book_id", dtype=DataType.INT64),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)  # Dimension depends on your embedding model
]
schema = CollectionSchema(fields, "Chunks collection")

# Create collection if it doesn't exist
if not utility.has_collection("chunks"):
    chunks_collection = Collection("chunks", schema)
    print("Created 'chunks' collection")
else:
    chunks_collection = Collection("chunks")
    print("'chunks' collection already exists")

# Create an index if it doesn't exist
if not chunks_collection.has_index():
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 1024}
    }
    chunks_collection.create_index(field_name="embedding", index_params=index_params)
    print("Created index on 'embedding' field")

#---------- The collection schema ---------------------
print("Collection schema:")
for field in chunks_collection.schema.fields:
    print(f"Field: {field.name}, Type: {field.dtype}, Is Primary: {field.is_primary}")

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def ingest_data(tree_index: TreeIndex):
    # Ensure the collection exists and is loaded
    chunks_collection = Collection("chunks")
    chunks_collection.load()

    batch_size = 100
    id_counter = 0

    ids = []
    chunk_ids = []
    section_ids = []
    book_ids = []
    contents = []
    embeddings = []

    for book_node in tree_index.get_nodes_by_type("book"):
        for section_node in book_node.children.values():
            for chunk_node in section_node.children.values():
                content = chunk_node.content
                if content:
                    try:
                        embedding = embedding_model.encode(content).tolist()
                        
                        ids.append(id_counter)
                        chunk_ids.append(int(chunk_node.id))
                        section_ids.append(int(section_node.id))
                        book_ids.append(int(book_node.id))
                        contents.append(content)
                        embeddings.append(embedding)
                        
                        id_counter += 1

                        # Insert in batches
                        if len(ids) == batch_size:
                            print(f"Inserting batch of {batch_size} entities")
                            chunks_collection.insert([
                                ids, chunk_ids, section_ids, book_ids, contents, embeddings
                            ])
                            ids, chunk_ids, section_ids, book_ids, contents, embeddings = [], [], [], [], [], []
                    except Exception as e:
                        print(f"Error processing chunk {chunk_node.id}: {str(e)}")

    # Insert any remaining entities
    if ids:
        print(f"Inserting final batch of {len(ids)} entities")
        chunks_collection.insert([
            ids, chunk_ids, section_ids, book_ids, contents, embeddings
        ])

    chunks_collection.flush()
    print(f"Ingested {id_counter} chunks into Milvus.")
# Usage
file_links = [
    'https://arxiv.org/pdf/2407.14562',
    'https://arxiv.org/pdf/2407.14743',
    'https://arxiv.org/pdf/2407.14662',
    'https://arxiv.org/pdf/2407.15259',
    'https://arxiv.org/pdf/2407.15527',
    'https://arxiv.org/pdf/2407.12873',
    'https://arxiv.org/pdf/2407.14525',
    'https://arxiv.org/pdf/2407.14565',
    'https://arxiv.org/pdf/2407.14568',
    'https://arxiv.org/pdf/2407.14622',
    'https://arxiv.org/pdf/2407.14631',
    'https://arxiv.org/pdf/2407.14651',
    'https://arxiv.org/pdf/2407.14658',
    'https://arxiv.org/pdf/2407.14681',
    'https://arxiv.org/pdf/2407.14717',
    'https://arxiv.org/pdf/2407.14725',
    'https://arxiv.org/pdf/2407.14735',
    'https://arxiv.org/pdf/2407.14738',
    'https://arxiv.org/pdf/2407.14741',
    'https://arxiv.org/pdf/2407.14765'
]
tree_index = process_files(file_links[:1])
ingest_data(tree_index)