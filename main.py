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
    embedding=HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh"),
    table="call_to_arms"
)

def index(file_name: str):
    """Indexes the content of a given file into the vector database."""
    loader = TextLoader(file_name)
    documents = loader.load()
    texts = chunk(documents)
    db.add_texts(texts)

def chunk(documents) -> list:
    """Splits documents into smaller chunks."""
    splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
    docs = splitter.split_documents(documents)
    texts = [doc.page_content for doc in docs]
    return texts

def is_indexed() -> bool:
    """Checks if the given document is indexed."""
    vec_db = Database("vec.db")
    table = vec_db["call_to_arms"]
    return table.count_where() > 0

SYSTEM_PROMPT = {
    "role": "system",
    "content": """你是一個強大的問題回答助理。
    你會收到下列格式的請求：
    Question: (question goes here)
    Context: (context goes here)
    Answer:
    根據 Context 回答問題，
    如果你不知道答案，就直接說你不知道。
    保持答案的簡潔，用最多三句話來回答。"""
}

def main():
    """Main function to run the RAG process."""
    if not is_indexed():
        index("romance-of-the-three-kingdoms.txt")

    messages = [SYSTEM_PROMPT]

    question = "孔明最大的貢獻是什麼？"
    docs = db.similarity_search(question)
    context = "".join([doc.page_content for doc in docs])
    messages.append({
        "role": "user",
        "content": f"""Question: {question}
        Context: {context}
        Answer:"""
    })

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-05-20",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    result = llm.invoke(messages)
    messages.append({
        "role": "assitant",
        "content": result.content
    })
    print(messages[-1]["content"])

if __name__ == "__main__":
    main()
