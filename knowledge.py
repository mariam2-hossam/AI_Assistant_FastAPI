import atexit
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, FieldCondition, Filter, FilterSelector,
                                  MatchValue, VectorParams)

load_dotenv()

COLLECTION = "knowledge"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)

# QDRANT_URL in .env = Qdrant server / cloud, otherwise it saves the data in the qdrant_db folder
if os.getenv("QDRANT_URL"):
    client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
else:
    client = QdrantClient(path="qdrant_db")

atexit.register(client.close)   # closes the qdrant folder cleanly when the program stops

if not client.collection_exists(COLLECTION):
    size = len(embeddings.embed_query("test"))   # size of the vectors, 384 for this model
    client.create_collection(COLLECTION, vectors_config=VectorParams(size=size, distance=Distance.COSINE))

vectorstore = QdrantVectorStore(client=client, collection_name=COLLECTION, embedding=embeddings)


def add_documents(docs):
    chunks = splitter.split_documents(docs)
    if not chunks:
        return 0
    # same text + same source = same id, so nothing gets added twice (qdrant ids must be uuids)
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, c.metadata["source"] + c.page_content)) for c in chunks]
    vectorstore.add_documents(chunks, ids=ids)
    return len(chunks)


def add_solution(question, solution):
    doc = Document(
        page_content="Question: " + question + "\n\nSolution:\n" + solution,
        metadata={"source": "user_feedback", "title": question[:80],
                  "question": question, "added_at": datetime.now().isoformat(timespec="seconds")},
    )
    return add_documents([doc])


def delete_source(source):
    # langchain keeps the metadata under the "metadata" key in the payload
    condition = FieldCondition(key="metadata.source", match=MatchValue(value=source))
    client.delete(COLLECTION, points_selector=FilterSelector(filter=Filter(must=[condition])))


def search(question, k=3):
    if client.count(COLLECTION).count == 0:
        return []
    return vectorstore.similarity_search(question, k=k)
