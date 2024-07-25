import json
import re
from typing import List, Dict
from langchain_community.document_loaders.llmsherpa import LLMSherpaFileLoader
from tree_indexer import TreeIndex

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
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(structure, f, ensure_ascii=False, indent=2)

def load_structure_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def print_structure(node, indent=0):
    print("  " * indent + f"[ID: {node['id']}] {node['type']}: {node['title']}")
    if node['content']:
        print("  " * (indent + 1) + f"Content: {node['content'][:50]}...")
    for child in node['children']:
        print_structure(child, indent + 1)

def process_files(file_links: List[str]):
    # Dictionary to store lists of dictionaries for each document
    all_documents = {}

    for i in range(len(file_links)):
        try:
            print(f"--------------------------Starting Data processing for item number: {i+1}-------------------------------------")
            print("preparing the loader")
            loader = LLMSherpaFileLoader(
                file_path=file_links[i],
                new_indent_parser=True,
                apply_ocr=True,
                strategy="sections",
                llmsherpa_api_url="http://localhost:5010/api/parseDocument?renderFormat=all"
            )
            
            print("loading the data")
            test_data = loader.load()
            print("loaded the data")

            # List of Document objects
            section_documents = test_data

            # Convert to list of dictionaries
            section_documents_dicts = []
            print("converting into dictionaries")
            for doc in section_documents:
                doc_dict = {
                    'metadata': doc.metadata,
                    'page_content': doc.page_content
                }
                section_documents_dicts.append(doc_dict)

            # Store the list of dictionaries in the main dictionary
            all_documents[f"document_{i+1}"] = section_documents_dicts
        except KeyError as e:
            print(f"Error processing {file_links[i]}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while processing {file_links[i]}: {e}")

    # Convert to library structure
    library_structure = convert_to_textbook_structure(all_documents.values(), max_chunk_size=500)

    # Save the structure to a JSON file
    json_file_path = "library_structure.json"
    save_structure_to_json(library_structure, json_file_path)

    print(f"Library structure saved to {json_file_path}")

    # Build the tree index
    tree_index = TreeIndex()
    tree_index.build_from_json(library_structure)

    return tree_index

if __name__ == "__main__":
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
    # process_files(file_links)