import json
import re
import logging
from typing import List, Dict
from argparse import ArgumentParser
from langchain_community.document_loaders.llmsherpa import LLMSherpaFileLoader
from tree_indexer import TreeIndex

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# ID-Generation--------------------

class IDGenerator:
    def __init__(self):
        self.current_id = 0

    def get_next_id(self):
        self.current_id += 1
        return self.current_id

id_generator = IDGenerator()

def create_node(title, node_type, content=None):
    return {
        "id": id_generator.get_next_id(),
        "title": title,
        "type": node_type,
        "children": [],
        "content": content if content else ""
    }

def chunk_content(content: str, max_chunk_size: int = 1000) -> List[str]:
    chunks = []
    current_chunk = ""
    sentences = re.split(r'(?<=[.!?])\s+', content)
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def convert_to_textbook_structure(books: List[List[Dict]], max_chunk_size: int = 1000):
    library = create_node("Library", "library")

    for book_index, book_documents in enumerate(books):
        book = create_node(f"Book {book_index + 1}", "book")
        library["children"].append(book)

        # Sort documents by section number
        sorted_documents = sorted(book_documents, key=lambda x: x['metadata'].get('section_number', 0))

        for doc in sorted_documents:
            section_title = doc['metadata'].get('section_title', 'No Title')
            content = doc['page_content']

            # Create a new section node
            new_section = create_node(section_title, "section")

            # Chunk the content
            content_chunks = chunk_content(content, max_chunk_size)

            # Create chunk nodes
            for i, chunk in enumerate(content_chunks):
                chunk_node = create_node(f"{section_title} - Chunk {i+1}", "chunk")
                chunk_node["content"] = chunk
                new_section["children"].append(chunk_node)

            # Add the new section to the book
            book["children"].append(new_section)

    return library

def save_structure_to_json(structure, file_path):
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(structure, f, ensure_ascii=False, indent=2)
        logging.info(f"Structure saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving structure to JSON: {e}")

def load_structure_from_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            logging.info(f"Loading structure from {file_path}")
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading structure from JSON: {e}")
        return None

def print_structure(node, indent=0):
    print("  " * indent + f"[ID: {node['id']}] {node['type']}: {node['title']}")
    if node['content']:
        print("  " * (indent + 1) + f"Content: {node['content'][:50]}...")
    for child in node['children']:
        print_structure(child, indent + 1)

def process_files(file_links: List[str]):
    all_documents = {}

    for i, file_link in enumerate(file_links):
        try:
            logging.info(f"Starting data processing for file: {file_link}")
            loader = LLMSherpaFileLoader(
                file_path=file_link,
                new_indent_parser=True,
                apply_ocr=True,
                strategy="sections",
                llmsherpa_api_url="http://localhost:5010/api/parseDocument?renderFormat=all"
            )
            logging.info("Loading data...")
            test_data = loader.load()
            logging.info("Data loaded.")

            section_documents_dicts = []
            for doc in test_data:
                doc_dict = {
                    'metadata': doc.metadata,
                    'page_content': doc.page_content
                }
                section_documents_dicts.append(doc_dict)

            all_documents[f"document_{i+1}"] = section_documents_dicts
        except KeyError as e:
            logging.error(f"KeyError processing {file_link}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error processing {file_link}: {e}")

    library_structure = convert_to_textbook_structure(all_documents.values(), max_chunk_size=500)
    json_file_path = "library_structure.json"
    save_structure_to_json(library_structure, json_file_path)
    logging.info(f"Library structure saved to {json_file_path}")

    tree_index = TreeIndex()
    tree_index.build_from_json(library_structure)

    return tree_index

if __name__ == "__main__":
    parser = ArgumentParser(description="Process PDF files and build a tree structure for indexing.")
    parser.add_argument("--file-links", type=str, nargs="+", required=True, help="List of file links to process.")

    args = parser.parse_args()

    tree_index = process_files(args.file_links)
    logging.info("Processing complete.")

# python components/data_prep.py --file-links "https://arxiv.org/pdf/2407.14562" "https://arxiv.org/pdf/2407.14743"
