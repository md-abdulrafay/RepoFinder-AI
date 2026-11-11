
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model():

    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    return embeddings


def create_vector_store(chunks, index_path="data/faiss_index"):

    embeddings = get_embedding_model()

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    vector_store.save_local(index_path)

    print(f"FAISS index saved at: {index_path}")

    return vector_store


def load_vector_store(index_path="data/faiss_index"):

    embeddings = get_embedding_model()

    vector_store = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store


def save_vector_store(vector_store, index_path="data/faiss_index"):

    vector_store.save_local(index_path)

    print(f"FAISS index saved at: {index_path}")
