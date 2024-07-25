import logging
from typing import Dict, List, Any
import re

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class TreeNode:
    def __init__(self, id: int, title: str, node_type: str):
        self.id = id
        self.title = title
        self.type = node_type
        self.children: Dict[str, TreeNode] = {}
        self.content: str = ""
        self.parent: TreeNode = None
        logging.debug(f"Initialized TreeNode: id={id}, title={title}, type={node_type}")

class TreeIndex:
    def __init__(self):
        self.root = TreeNode(0, "Root", "root")
        self.id_map: Dict[int, TreeNode] = {0: self.root}
        self.type_map: Dict[str, List[TreeNode]] = {
            "library": [],
            "book": [],
            "section": [],
            "chunk": []
        }
        logging.debug("TreeIndex initialized with root node.")

    def add_node(self, node_data: Dict[str, Any], parent_id: int):
        parent = self.id_map[parent_id]
        logging.debug(f"Adding node with data: {node_data} to parent ID: {parent_id}")

        new_node = TreeNode(node_data["id"], node_data["title"], node_data["type"])
        new_node.content = node_data.get("content", "")
        new_node.parent = parent
        
        parent.children[new_node.title.lower()] = new_node
        self.id_map[new_node.id] = new_node
        self.type_map[new_node.type].append(new_node)

        logging.info(f"Node added: id={new_node.id}, title={new_node.title}, type={new_node.type}")

        for child in node_data.get("children", []):
            self.add_node(child, new_node.id)

    def build_from_json(self, json_data: Dict[str, Any]):
        logging.debug("Building tree from JSON data.")
        self.add_node(json_data, 0)
        logging.info("Tree building complete.")

    def search(self, query: str) -> List[TreeNode]:
        words = re.findall(r'\w+', query.lower())
        results = []
        logging.debug(f"Search query: {query} -> words: {words}")
        
        def dfs(node: TreeNode, depth: int):
            if depth == len(words):
                results.append(node)
                logging.debug(f"Found match at node: {node.id} - {node.title}")
                return
            
            for child_title, child_node in node.children.items():
                if words[depth] in child_title.lower():
                    dfs(child_node, depth + 1)
        
        dfs(self.root, 0)
        logging.info(f"Search completed. Found {len(results)} results.")
        return results

    def get_node_by_id(self, node_id: int) -> TreeNode:
        node = self.id_map.get(node_id)
        if node:
            logging.debug(f"Node found by ID {node_id}: {node.title}")
        else:
            logging.warning(f"Node with ID {node_id} not found.")
        return node

    def get_nodes_by_type(self, node_type: str) -> List[TreeNode]:
        nodes = self.type_map.get(node_type, [])
        logging.debug(f"Nodes found by type '{node_type}': {len(nodes)} nodes.")
        return nodes

    def get_path(self, node: TreeNode) -> List[str]:
        path = []
        current = node
        while current != self.root:
            path.append(current.title)
            current = current.parent
        logging.debug(f"Path for node {node.id}: {' > '.join(reversed(path))}")
        return list(reversed(path))

    def filter_sections(self, query: str) -> List[int]:
        results = self.search(query)
        section_ids = []
        for node in results:
            if node.type == "section":
                section_ids.append(node.id)
            elif node.type == "chunk":
                section_node = node.parent
                section_ids.append(section_node.id)
        section_ids = list(set(section_ids))  # Remove duplicates
        logging.debug(f"Filtered sections IDs: {section_ids}")
        return section_ids

    def get_chunk_info(self, chunk_id: int) -> Dict[str, Any]:
        chunk_node = self.get_node_by_id(chunk_id)
        if not chunk_node or chunk_node.type != "chunk":
            logging.warning(f"Chunk with ID {chunk_id} not found or not a chunk type.")
            return None
        
        section_node = chunk_node.parent
        book_node = section_node.parent
        
        chunk_info = {
            "chunk_id": chunk_id,
            "content": chunk_node.content,
            "section_title": section_node.title,
            "book_title": book_node.title
        }
        logging.debug(f"Chunk info retrieved for ID {chunk_id}: {chunk_info}")
        return chunk_info
