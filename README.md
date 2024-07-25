Here's the entire README.md content in a single code block:


# T_RAG Setup Guide

This guide will walk you through setting up the T_RAG system, including data scraping, database setup, and query processing. The system uses Milvus for vector search, Elasticsearch for BM25, and `nlm-ingestor` for document layout and PDF parsing.

## Clone the Repository and Install Dependencies

First, clone the repository and install the required Python packages.

```bash
git clone https://github.com/KripaMishra/T_RAG.git
cd T_RAG/
pip install -r requirements.txt
```

Ensure that all dependencies are properly installed before proceeding to the next steps.


## Setting up Milvus and Elasticsearch

**Note**: While Elasticsearch isn't required immediately, it's recommended to set up all components simultaneously for a hybrid indexing approach.

### Milvus Setup

Milvus is used for managing the vector database.

1. **Install Docker**:

    ```bash
    sudo apt-get install docker.io
    ```

2. **Download and start the Milvus installation script**:

    ```bash
    curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
    bash standalone_embed.sh start
    ```

3. **Stop or delete Milvus**:

    To stop Milvus:
    ```bash
    bash standalone_embed.sh stop
    ```

    To delete Milvus data:
    ```bash
    bash standalone_embed.sh delete
    ```

### Elasticsearch Setup

Elasticsearch is used for BM25 text search. Follow these steps to install and set up Elasticsearch:

1. **Download and verify Elasticsearch for Linux**:

    ```bash
    wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.14.3-linux-x86_64.tar.gz
    wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.14.3-linux-x86_64.tar.gz.sha512
    shasum -a 512 -c elasticsearch-8.14.3-linux-x86_64.tar.gz.sha512
    ```

2. **Extract the downloaded archive**:

    ```bash
    tar -xzf elasticsearch-8.14.3-linux-x86_64.tar.gz
    cd elasticsearch-8.14.3/
    ```

3. **Enable automatic creation of system indices** (optional):

    Edit `config/elasticsearch.yml` to include:
    ```yaml
    action.auto_create_index: .monitoring*,.watches,.triggered_watches,.watcher-history*,.ml*
    ```

4. **Run Elasticsearch**:

    ```bash
    ./bin/elasticsearch
    ```

### NLM Ingestor Setup

The `nlm-ingestor` service is used for document layout and PDF parsing. Set it up with the following commands:

1. **Pull and run the NLM Ingestor Docker image**:

    ```bash
    docker pull ghcr.io/nlmatics/nlm-ingestor:latest
    docker run -p 5010:5001 ghcr.io/nlmatics/nlm-ingestor:latest
    ```

## Data Ingestion

After setting up Milvus, Elasticsearch, and `nlm-ingestor`, you can ingest data into the database collection:

```bash
python components/data_ingestion.py --file-links "https://arxiv.org/pdf/2407.14562" "https://arxiv.org/pdf/2407.14743"
```

## Query Processing

To generate answers for specific queries, use the following command:

```bash
python components/RAG.py "what are the major obstacles for output generation on a semantic level?"
```

Ensure that all services (Milvus, Elasticsearch, and `nlm-ingestor`) are running before executing these commands.


