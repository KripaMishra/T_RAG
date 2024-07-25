import logging
from argparse import ArgumentParser
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, connections, utility
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from tree_indexer import TreeIndex
from data_prep import *

import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def connect_to_services(milvus_host, milvus_port, es_url):
    try:
        # Connect to Milvus
        connections.connect(host=milvus_host, port=milvus_port)
        logging.info(f"Connected to Milvus at {milvus_host}:{milvus_port}")
    except Exception as e:
        logging.error(f"Failed to connect to Milvus: {str(e)}")
        raise e
    
    try:
        # Connect to Elasticsearch
        es = Elasticsearch(es_url)
        logging.info(f"Connected to Elasticsearch at {es_url}")
        return es
    except Exception as e:
        logging.error(f"Failed to connect to Elasticsearch: {str(e)}")
        raise e

def setup_collection():
    try:
        # Define Milvus collection schema
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
        if not utility.has_collection("Test_Collection"):
            chunks_collection = Collection("Test_Collection", schema)
            logging.info("Created 'Test Collection' collection")
        else:
            chunks_collection = Collection("Test_Collection")
            logging.info("'Test_Collection' collection already exists")

        # Create an index if it doesn't exist
        if not chunks_collection.has_index():
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "params": {"nlist": 1024}
            }
            chunks_collection.create_index(field_name="embedding", index_params=index_params)
            logging.info("Created index on 'embedding' field")
        
        return chunks_collection
    except Exception as e:
        logging.error(f"Error setting up collection: {str(e)}")
        raise e

def ingest_data(tree_index: TreeIndex, chunks_collection, es):
    try:
        # Ensure the collection exists and is loaded
        chunks_collection.load()
        logging.info("Loaded 'Test_Collection' into memory.")

        batch_size = 100
        id_counter = 0

        ids = []
        chunk_ids = []
        section_ids = []
        book_ids = []
        contents = []
        embeddings = []

        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

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
                            
                            # Use the same ID for both Milvus and Elasticsearch
                            es.index(index="tree_data", id=id_counter, body={
                                "text": content,
                                "chunk_id": chunk_node.id,
                                "section_id": section_node.id,
                                "book_id": book_node.id
                            })
                            
                            id_counter += 1

                            # Insert in batches
                            if len(ids) == batch_size:
                                logging.info(f"Inserting batch of {batch_size} entities")
                                chunks_collection.insert([
                                    ids, chunk_ids, section_ids, book_ids, contents, embeddings
                                ])
                                ids, chunk_ids, section_ids, book_ids, contents, embeddings = [], [], [], [], [], []
                        except Exception as e:
                            logging.error(f"Error processing chunk {chunk_node.id}: {str(e)}")

        # Insert any remaining entities
        if ids:
            logging.info(f"Inserting final batch of {len(ids)} entities")
            chunks_collection.insert([
                ids, chunk_ids, section_ids, book_ids, contents, embeddings
            ])

        chunks_collection.flush()
        logging.info(f"Ingested {id_counter} chunks into Milvus.")
    except Exception as e:
        logging.error(f"Error during data ingestion: {str(e)}")
        raise e

def main(milvus_host, milvus_port, es_url, file_links):
    try:
        es = connect_to_services(milvus_host, milvus_port, es_url)
        chunks_collection = setup_collection()
        tree_index = process_files(file_links)
        ingest_data(tree_index, chunks_collection, es)
    except Exception as e:
        logging.critical(f"Fatal error in main: {str(e)}")
        raise e

# sample data:

# file_links = [
#     'https://arxiv.org/pdf/2407.14562',
#     'https://arxiv.org/pdf/2407.14743',
#     'https://arxiv.org/pdf/2407.14662',
#     'https://arxiv.org/pdf/2407.15259',
#     'https://arxiv.org/pdf/2407.15527',
#     'https://arxiv.org/pdf/2407.12873',
#     'https://arxiv.org/pdf/2407.14525',
#     'https://arxiv.org/pdf/2407.14565',
#     'https://arxiv.org/pdf/2407.14568',
#     'https://arxiv.org/pdf/2407.14622',
#     'https://arxiv.org/pdf/2407.14631',
#     'https://arxiv.org/pdf/2407.14651',
#     'https://arxiv.org/pdf/2407.14658',
#     'https://arxiv.org/pdf/2407.14681',
#     'https://arxiv.org/pdf/2407.14717',
#     'https://arxiv.org/pdf/2407.14725',
#     'https://arxiv.org/pdf/2407.14735',
#     'https://arxiv.org/pdf/2407.14738',
#     'https://arxiv.org/pdf/2407.14741',
#     'https://arxiv.org/pdf/2407.14765'
# ]

if __name__ == "__main__":
    parser = ArgumentParser(description="Ingest data into Milvus and Elasticsearch.")
    parser.add_argument("--milvus-host", type=str, default="localhost", help="Milvus server host.")
    parser.add_argument("--milvus-port", type=int, default=19530, help="Milvus server port.")
    parser.add_argument("--es-url", type=str, default="http://localhost:9200", help="Elasticsearch URL.")
    parser.add_argument("--file-links", type=str, nargs="+", help="List of file links to process.")

    args = parser.parse_args()

    main(args.milvus_host, args.milvus_port, args.es_url, args.file_links)



# python components/data_ingestion.py  --file-links "https://arxiv.org/pdf/2407.14562" "https://arxiv.org/pdf/2407.14743"
