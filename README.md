# RAG Implementation

This project demonstrates a simple Retrieval-Augmented Generation (RAG) implementation.

## Features

- **Text Chunking**: Splits large documents into smaller, manageable chunks.
- **Chunk Indexing**: Indexes chunks into a SQLite vector database.
- **Similarity Search**: Performs similarity searches on the indexed chunks to retrieve relevant context.
- **Question Answering**: Uses a large language model (LLM) to answer questions based on the retrieved context.

## Setup

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd rag
    ```

2.  **Install dependencies**:
    It is recommended to use `uv` for dependency management.

    ```bash
    uv sync
    ```

3.  **Prepare your data**:
    Place your text document (e.g., `romance-of-the-three-kingdoms.txt`) in the project root directory. The `main.py` script is configured to index `romance-of-the-three-kingdoms.txt` by default.

4.  **Set up Google API Key**:
    Create a `.env` file in the project root and add your Google API Key:

    ```
    GOOGLE_API_KEY="your_google_api_key_here"
    ```

## Usage

To run the RAG process and get an answer to a predefined question:

```bash
uv run main.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
