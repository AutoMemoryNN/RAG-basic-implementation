import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

loader = PyPDFLoader("test.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100,)
chunks = splitter.split_documents(pages)

# 2. Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

qdrant = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    url=QDRANT_URL,
    prefer_grpc=True,
    api_key=QDRANT_API_KEY,
    collection_name="Test_Cluster",
    force_recreate=True, # delete in production
)

# 4. Consultas
retriever = qdrant.as_retriever(
    search_type="similarity_score_threshold", # can be "similarity_score_threshold" or "similarity" or "mmr"
    search_kwargs={
        "score_threshold": 0.9,  # solo retorna docs con score ≥ 0.9
        "k": 5                    # máximo 5 documentos
    }
)
docs = retriever.get_relevant_documents("¿Cual es el nombre del curso?")
for d in docs:
    print("===============")
    print(d.metadata)
    print("\n")
    print(d.page_content)
    print("===============")
    print("\n")