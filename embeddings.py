from langchain_community.embeddings import HuggingFaceEmbeddings
import joblib

EMBEDDINGS_PATH = 'vectorstore/embeddings.joblib'

def save_embeddings(embeddings):
    joblib.dump(embeddings, EMBEDDINGS_PATH)
    print(f"Embeddings saved to {EMBEDDINGS_PATH}")

def load_embeddings():
    print(f"Loading embeddings from {EMBEDDINGS_PATH}")
    embeddings = joblib.load(EMBEDDINGS_PATH)
    print("Embeddings loaded successfully")
    return embeddings
