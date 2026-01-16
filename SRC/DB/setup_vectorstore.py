from chromadb import PersistentClient
from chromadb.utils import embedding_functions
import json
import os

def create_vectorstore(collection_name: str, persist_directory: str, embedding_function) -> None:
    """
    Create and return a ChromaDB vector store client with a specified collection.
    """

    client = PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(name=collection_name)
    datasest_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../DATAS/dataset.json")
    with open(datasest_path,'r') as f:
        dataset = json.load(f)
    
    documents = [f"Diagnosis: {row['diagnosis']}\nSymptoms: {row['symptoms']}\nTreatment: {row['treatment']}" for row in dataset]
    ids = [str(i) for i in range(len(documents))]
    embeddings = [embedding_function(doc) for doc in documents]
    collection.add(documents=documents, embeddings=embeddings, ids=ids)

if __name__ == "__main__":  
    vectorstore_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../chroma_persist")
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    vectorstore_client = create_vectorstore(
        collection_name="blockcure_collection",
        persist_directory=vectorstore_path,
        embedding_function=embedding_function
    )