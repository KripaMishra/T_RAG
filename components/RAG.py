# rag_model.py
import json
import argparse
from datetime import datetime
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from data_retriever import hybrid_search  # Import hybrid_search function from data_retriever.py

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RAGModel:
    def __init__(self, model_name: str):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.max_length = self.model.config.max_position_embeddings
            logging.info("RAGModel initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing RAGModel: {e}")
            raise

    def generate_answer(self, query: str) -> str:
        try:
            # Perform retrieval
            retrieval_results = hybrid_search(query)

            # Construct the context for the language model
            context = f"Original query: {query}\n"
            context += "Retrieved documents:\n"
            for result in retrieval_results:
                context += f"- {result['content'][:200]}...\n"  # Limit to first 200 characters

            # Generate answer using the language model
            prompt = (f"{context}\nBased on the information above, provide a concise and informative response to the query.")

            # Tokenize the input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length)
            logging.info(f"Token IDs: {inputs.input_ids}")
            logging.info(f"Token IDs shape: {inputs.input_ids.shape}")

            # Generate answer
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=100,
                    num_return_sequences=1,
                    temperature=0.10,
                    do_sample=True,
                    top_k=25,
                    top_p=0.75,
                    no_repeat_ngram_size=3,
                )
            
            # Decode the generated answer
            answer = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return answer.strip()
        except Exception as e:
            logging.error(f"Error generating answer: {e}")
            return "An error occurred while generating the answer."

    def process_query(self, query: str) -> dict:
        try:
            timestamp = datetime.now().isoformat()
            answer = self.generate_answer(query)

            return {
                "timestamp": timestamp,
                "original_query": query,
                "answer": answer,
            }
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "original_query": query,
                "error": str(e)
            }

    def save_query_results(self, query_data, file_path=None):
        """
        Save the query information and results as a JSON file.
        
        :param query_data: Dictionary containing query information and results
        :param file_path: Optional file path. If None, a default path will be used.
        """
        try:
            # Generate a default file path if none is provided
            if file_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = f"result/query_results_{timestamp}.json"
            
            # Save the data to a JSON file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(query_data, f, ensure_ascii=False, indent=4)
            
            logging.info(f"Query results saved to {file_path}")
        except Exception as e:
            logging.error(f"Error saving query results: {e}")

def main(query, file_path):
    model_name = "gpt2-xl"  # Replace with your preferred model
    rag_model = RAGModel(model_name)
    
    result = rag_model.process_query(query)
    rag_model.save_query_results(result, file_path)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Query Retrieval and Generation')
    parser.add_argument('query', type=str, help='The query string')
    parser.add_argument('--file_path', type=str, default=None, help='Path to save query results')
    args = parser.parse_args()
    main(args.query, args.file_path)
