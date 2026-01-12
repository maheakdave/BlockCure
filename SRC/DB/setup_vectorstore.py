from chromadb import PersistentClient
from chromadb.utils import embedding_functions
import json


def create_vectorstore(collection_name: str, persist_directory: str, embedding_function) -> None:
    """
    Create and return a ChromaDB vector store client with a specified collection.
    """

    client = PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(name=collection_name)

    with open('dataset.json','r') as f:
        dataset = json.load(f)
    
    documents = [f"Diagnosis: {row['diagnosis']}\nSymptoms: {row['symptoms']}\nTreatment: {row['treatment']}" for row in dataset]
    ids = [str(i) for i in range(len(documents))]
    embeddings = [embedding_function(doc) for doc in documents]
    collection.add(documents=documents, embeddings=embeddings, ids=ids)

if __name__ == "__main__":  
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    vectorstore_client = create_vectorstore(
        collection_name="blockcure_collection",
        persist_directory="../../chroma_persist",
        embedding_function=embedding_function
    )