# from pymilvus import Collection, connections
# from sentence_transformers import SentenceTransformer
# from data_prep import *
# from tree_indexer import
# # Connect to Milvus
# connections.connect(host='localhost', port='19530')

# # Initialize collections
# chunks_collection = Collection("chunks")
# chunks_collection.load()

# # Initialize the embedding model
# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# def hybrid_search(query: str, tree_index: TreeIndex, top_k: int = 5):
#     # 1. Keyword-based filtering
#     relevant_section_ids = tree_index.filter_sections(query)
    
#     # 2. Vector similarity search
#     query_vector = embedding_model.encode(query).tolist()
#     search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
#     results = chunks_collection.search(
#         data=[query_vector],
#         anns_field="embedding",
#         param=search_params,
#         limit=top_k,
#         expr=f"section_id in {relevant_section_ids}"
#     )

#     # 3. Result combination and context enrichment
#     combined_results = []
#     for hits in results:
#         for hit in hits:
#             chunk_id = hit.id
#             chunk_info = tree_index.get_chunk_info(chunk_id)
#             if chunk_info:
#                 enriched_result = {
#                     "chunk_id": chunk_id,
#                     "content": chunk_info["content"],
#                     "section_title": chunk_info["section_title"],
#                     "book_title": chunk_info["book_title"],
#                     "similarity_score": hit.distance,
#                     "path": tree_index.get_path(tree_index.get_node_by_id(chunk_id))
#                 }
#                 combined_results.append(enriched_result)
    
#     return combined_results

# # Example usage
# tree_index = process_files(file_links)
# results = hybrid_search("quantum computing applications", tree_index)
# for result in results:
#     print(f"Book: {result['book_title']}")
#     print(f"Section: {result['section_title']}")
#     print(f"Path: {' > '.join(result['path'])}")
#     print(f"Content: {result['content'][:100]}...")
#     print(f"Score: {result['similarity_score']}")
#     print("---")




from pymilvus import Collection, connections
from sentence_transformers import SentenceTransformer
import json
from datetime import datetime

# Connect to Milvus
connections.connect(host='localhost', port='19530')

# Initialize collections
chunks_collection = Collection("chunks")
chunks_collection.load()

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def hybrid_search(query: str, top_k: int = 5):
    # Vector similarity search
    query_vector = embedding_model.encode(query).tolist()
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = chunks_collection.search(
        data=[query_vector],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["chunk_id", "section_id", "book_id", "content"]
    )

    # Result processing
    combined_results = []
    for hits in results:
        for hit in hits:
            result = {
                "id": str(hit.id),  # Convert to string for JSON serialization
                "chunk_id": hit.entity.get("chunk_id"),
                "section_id": hit.entity.get("section_id"),
                "book_id": hit.entity.get("book_id"),
                "content": hit.entity.get("content"),
                "similarity_score": hit.distance
            }
            combined_results.append(result)
    
    return combined_results

def save_results_to_json(query: str, results: list):
    # Create a dictionary with query and results
    data = {
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "results": results
    }
    
    # Generate a filename based on the current timestamp
    filename = f"retriever_results/search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Save the data to a JSON file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print(f"Results saved to {filename}")

# Example usage
query = " we      employ      the      Prolog      engine      to      perform logical      reasoning"
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
    print("---")