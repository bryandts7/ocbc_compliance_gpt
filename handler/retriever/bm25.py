from langchain_community.retrievers import BM25Retriever
from constants.docs import REKAM_DOCS, KETENTUAN_DOCS 

def bm25_rekam_retriever():
    return rekam_retriever

def bm25_ketentuan_retriever():
    return ketentuan_retriever

rekam_retriever = BM25Retriever.from_documents(REKAM_DOCS)
ketentuan_retriever = BM25Retriever.from_documents(KETENTUAN_DOCS)


