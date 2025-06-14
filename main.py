"""Module for the implementation of RAG."""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import SQLiteVec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import CharacterTextSplitter
from sqlite_utils import Database

load_dotenv()

db = SQLiteVec(
    connection=SQLiteVec.create_connection(db_file="vec.db"),
    embedding=HuggingFaceEmbeddings(model_name="BAAI/bge-m3"),
    table="romance_of_the_three_kingdoms"
)

def index(file_name: str):
    """Indexes the content of a given file into the vector database."""
    loader = TextLoader(file_name)
    documents = loader.load()
    texts = chunk(documents)
    db.add_texts(texts)

def chunk(documents) -> list:
    """Splits documents into smaller chunks."""
    splitter = CharacterTextSplitter(chunk_size=256, chunk_overlap=0)
    docs = splitter.split_documents(documents)
    texts = [doc.page_content for doc in docs]
    return texts

def is_indexed() -> bool:
    """Checks if the given document is indexed."""
    vec_db = Database("vec.db")
    table = vec_db["romance_of_the_three_kingdoms"]
    return table.count_where() > 0

SYSTEM_PROMPT = {
    "role": "system",
    "content": """You are a powerful question-answering assistant.
    You will receive requests in the following format:
    Question: (question goes here)
    Context: (context goes here)
    Answer:
    Answer the question based on the Context.
    If you do not know the answer, simply state that you do not know.
    Keep the answer concise, using a maximum of three sentences."""
}

def main():
    """Main function to run the RAG process."""
    if not is_indexed():
        index("romance-of-the-three-kingdoms.txt")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-05-20",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    messages = [SYSTEM_PROMPT]

    while True:
        print("Ask me anything.")
        question = input()
        docs = db.similarity_search(question)
        context = "".join([doc.page_content for doc in docs])
        messages.append({
            "role": "user",
            "content": f"""Question: {question}
            Context: {context}
            Answer:"""
        })

        result = llm.invoke(messages)
        messages.append({
            "role": "assistant",
            "content": result.content
        })
        print(f"{messages[-1]["content"]}\n")

if __name__ == "__main__":
    main()
