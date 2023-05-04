# pyvsearch

This project provides a vector search server using the gRPC framework. The server is built using the FAISS library for efficient similarity search and leveldb for persistent storage. It is designed to handle a variety of index types and configurations. The server allows for upserting, training, deleting, searching, and retrieving database statistics.

## Features

- Supports multiple index types: IndexFlatL2, IndexIVFFlat, IndexIVFPQ
- Customizable index parameters like quantizer type, nlist, and M (subquantizers)
- Supports training for non-flat index types
- Handles multiple namespaces for separate vector spaces
- Efficient search using the FAISS library
- Persistent storage using leveldb

## Requirements

- Python 3.8 or higher
- FAISS library
- LevelDB
- gRPC
- NumPy

If you use macOS, `brew install leveldb` version 1.22.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/ground0state/pyvsearch.git
    ```

2. Create a virtual environment and activate it:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

To start the server, run the following command:

```bash
python server.py
```

The server will start on port 50051. You can interact with the server using a gRPC client.

## API

The server exposes the following gRPC services:

- Upsert: Add or update vector data to the index.
- Train: Train the index if it requires training (e.g., non-flat index types).
- DeleteAll: Remove all data from the index and reset it.
- Delete: Remove specific data points from the index using their IDs.
- Search: Search for similar vectors in the index.
- DbStats: Retrieve database statistics and configuration.

Please refer to the `vsearch_pb2.py` and `vsearch_pb2_grpc.py` files for detailed information on the service definitions and request/response messages.

## TODO

- [ ] Logging
- [ ] Authentication
- [ ] TLS
- [ ] Packaging
- [ ] Client SDK
- [ ] Multi index
