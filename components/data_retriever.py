import json
import logging
from datetime import datetime
from typing import List
from pymilvus import Collection, connections
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Connect to Milvus
connections.connect(host='localhost', port='19530')

# Connect to Elasticsearch
es = Elasticsearch("http://localhost:9200")

# Initialize collections
chunks_collection = Collection("Test_Collection")
chunks_collection.load()

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def normalize_scores(scores: List[float]) -> List[float]:
    """Normalize a list of scores to a range of 0 to 1."""
    min_score = min(scores)
    max_score = max(scores)
    if max_score - min_score == 0:
        return [1.0] * len(scores)  # Prevent division by zero if all scores are the same
    return [(score - min_score) / (max_score - min_score) for score in scores]

def hybrid_search(query: str, top_k: int = 5) -> List[dict]:
    """Perform a hybrid search using Milvus and Elasticsearch."""
    try:
        # Vector similarity search in Milvus
        query_vector = embedding_model.encode(query).tolist()
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        milvus_results = chunks_collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["chunk_id", "section_id", "book_id", "content"]
        )

        # Process Milvus results
        milvus_hits, milvus_scores = [], []
        for hits in milvus_results:
            for hit in hits:
                similarity_score = 1 / (1 + hit.distance)  # Transform distance to similarity score
                milvus_scores.append(similarity_score)
                result = {
                    "id": str(hit.id),
                    "chunk_id": hit.entity.get("chunk_id"),
                    "section_id": hit.entity.get("section_id"),
                    "book_id": hit.entity.get("book_id"),
                    "content": hit.entity.get("content"),
                    "similarity_score": similarity_score,
                    "source": "milvus"
                }
                milvus_hits.append(result)

        # Normalize Milvus scores
        milvus_scores = normalize_scores(milvus_scores)
        for i, hit in enumerate(milvus_hits):
            hit["similarity_score"] = milvus_scores[i]

        # Text search in Elasticsearch
        es_results = es.search(
            index="tree_data",
            body={
                "query": {
                    "match": {
                        "text": query
                    }
                },
                "size": top_k
            }
        )

        # Process Elasticsearch results
        es_hits, es_scores = [], []
        for hit in es_results['hits']['hits']:
            es_scores.append(hit["_score"])
            es_hit = {
                "id": hit["_id"],
                "chunk_id": hit["_source"]["chunk_id"],
                "section_id": hit["_source"]["section_id"],
                "book_id": hit["_source"]["book_id"],
                "content": hit["_source"]["text"],
                "similarity_score": hit["_score"],
                "source": "elasticsearch"
            }
            es_hits.append(es_hit)

        # Normalize Elasticsearch scores
        es_scores = normalize_scores(es_scores)
        for i, hit in enumerate(es_hits):
            hit["similarity_score"] = es_scores[i]

        # Combine results
        combined_results = milvus_hits + es_hits

        # Sort combined results by similarity_score (higher is better)
        combined_results.sort(key=lambda x: x["similarity_score"], reverse=True)

        # Return top_k results
        return combined_results[:top_k]
    except Exception as e:
        logging.error(f"An error occurred during the hybrid search: {e}")
        return []

def save_results_to_json(query: str, results: List[dict]):
    """Save search results to a JSON file."""
    try:
        data = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "results": results
        }

        filename = f"retriever_results/search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logging.info(f"Results saved to {filename}")
    except Exception as e:
        logging.error(f"Failed to save results to JSON: {e}")

if __name__ == "__main__":
    # Example usage
    query = "we employ the Prolog engine to perform logical reasoning"
    results = hybrid_search(query)

    # Save results to JSON file
    save_results_to_json(query, results)

    # Print results to console
    for result in results:
        print(f"ID: {result['id']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Book ID: {result['book_id']}")
        print(f"Section ID: {result['section_id']}")
        print(f"Content: {result['content'][:100]}...")
        print(f"Score: {result['similarity_score']}")
        print(f"Source: {result['source']}")
        print("---")
