import json
import argparse
from datetime import datetime
import logging
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from data_retriever import hybrid_search
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Setup logging (unchanged)
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"rag_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

class RAGModel:
    def __init__(self, model_name: str):
        try:
            logging.info(f"Initializing RAGModel with model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.max_length = self.model.config.max_position_embeddings
            self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
            logging.info("RAGModel initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing RAGModel: {e}", exc_info=True)
            raise

    def generate_answer(self, query: str) -> tuple:
        try:
            logging.info(f"Generating answer for query: {query}")
            
            logging.info("Performing hybrid search...")
            retrieval_results = hybrid_search(query)
            logging.info(f"Retrieved {len(retrieval_results)} documents")

            context = f"Query: {query}\n\nRelevant information:\n"
            for i, result in enumerate(retrieval_results, 1):
                context += f"{i}. {result['content'][:100].strip()}...\n"
            
            prompt = f"{context}\n\nBased on the above information, please provide a concise and relevant answer to the query. If the information is not sufficient to answer the query, state that clearly.\n\nAnswer:"

            # Ensure the prompt fits within the model's maximum length
            encoded_prompt = self.tokenizer.encode(prompt, return_tensors="pt")
            max_length = min(self.max_length, 1024)  # Use the smaller of model's max length or 1024
            if encoded_prompt.size(1) > max_length - 200:  # Reserve 100 tokens for the answer
                encoded_prompt = encoded_prompt[:, :max_length - 200]
                prompt = self.tokenizer.decode(encoded_prompt[0], skip_special_tokens=True)

            logging.info("Generating answer...")
            response = self.generator(prompt, max_new_tokens=200, num_return_sequences=1, temperature=0.5, top_k=50, top_p=0.95, do_sample=True)
            
            answer = response[0]['generated_text'][len(prompt):].strip()
            logging.info("Answer generated successfully")
            return answer, retrieval_results
        except Exception as e:
            logging.error(f"Error generating answer: {e}", exc_info=True)
            return "An error occurred while generating the answer.", []

    def process_query(self, query: str) -> dict:
        try:
            logging.info(f"Processing query: {query}")
            timestamp = datetime.now().isoformat()
            answer, retrieved_docs = self.generate_answer(query)

            result = {
                "timestamp": timestamp,
                "original_query": query,
                "retrieved_documents": retrieved_docs,
                "answer": answer
            }
            logging.info("Query processed successfully")
            return result
        except Exception as e:
            logging.error(f"Error processing query: {e}", exc_info=True)
            return {
                "timestamp": datetime.now().isoformat(),
                "original_query": query,
                "error": str(e)
            }

    # save_query_results method remains unchanged
    def save_query_results(self, query_data, file_path=None):
        try:
            if file_path is None:
                output_dir = "generated_output"
                os.makedirs(output_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = os.path.join(output_dir, f"query_results_{timestamp}.json")
            
            logging.info(f"Saving query results to: {file_path}")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(query_data, f, ensure_ascii=False, indent=4)
            
            logging.info(f"Query results saved successfully")
        except Exception as e:
            logging.error(f"Error saving query results: {e}", exc_info=True)

def main(query, file_path):
    try:
        logging.info("Starting main function")
        model_name = "google/gemma-1.1-2b-it"  # Replace with your preferred model
        logging.info(f"Initializing RAGModel with model: {model_name}")
        rag_model = RAGModel(model_name)
        
        logging.info("Processing query")
        result = rag_model.process_query(query)
        
        logging.info("Saving query results")
        rag_model.save_query_results(result, file_path)
        
        logging.info("Printing results")
        print(json.dumps(result, indent=2))
        
        logging.info("Main function completed successfully")
    except Exception as e:
        logging.error(f"Error in main function: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Query Retrieval and Generation')
    parser.add_argument('query', type=str, help='The query string')
    parser.add_argument('--file_path', type=str, default=None, help='Path to save query results')
    args = parser.parse_args()
    main(args.query, args.file_path)

#  python components/RAG.py "what are the major obstacles for output generation on semantic level ?"

