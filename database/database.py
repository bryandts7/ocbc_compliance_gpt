from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders.mongodb import MongodbLoader

from utils.azure_openai import azure_embeddings

import time
import os
import dotenv
dotenv.load_dotenv()

embeddings = azure_embeddings()

# Pinecone Database
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

# MongoDB Database
mongo_uri = os.environ.get("MONGO_URI")
REKAM_JEJAK_DOCSTORE = MongodbLoader(
    connection_string= mongo_uri, # type: ignore
    db_name="docstore",
    collection_name="rekam_jejak",
)
KETENTUAN_TERKAIT_DOCSTORE = MongodbLoader(
    connection_string= mongo_uri, # type: ignore
    db_name="docstore",
    collection_name="ketentuan_terkait",
)

def rekam_jejak_vector():
    index_name = "rekam-jejak"
    index = pc.Index(index_name)
    rekam_jejak_vector = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    return rekam_jejak_vector

def rekam_jejak_docstore():
    return REKAM_JEJAK_DOCSTORE

def ketentuan_terkait_vector():
    index_name = "ketentuan-terkait"
    index = pc.Index(index_name)
    ketentuan_terkait_vector = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    return ketentuan_terkait_vector

def ketentuan_terkait_docstore():
    return KETENTUAN_TERKAIT_DOCSTORE