import gzip
import pickle

from langchain.retrievers import (ContextualCompressionRetriever,
                                  MergerRetriever)
from langchain.retrievers.document_compressors.base import \
    DocumentCompressorPipeline
from langchain_cohere import CohereRerank
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_community.retrievers import BM25Retriever
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.vectorstores import VectorStore

all_documents_file = gzip.open(f'retriever/retriever_bi/all_documents.pkl.gz','rb')
all_documents = pickle.load(all_documents_file)

def get_retriever_bi(vector_store: VectorStore, llm_model: BaseLanguageModel, embed_model: Embeddings, top_n: int = 7, top_k:int = 20, config: dict = {}):

    top_k = top_k // 3

    retriever_similarity = vector_store.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": top_k}
    )
    retriever_mmr = vector_store.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": top_k}
    )
    retriever_bm25 = BM25Retriever.from_documents(all_documents, k=top_k)


    # merge retrievers
    lotr = MergerRetriever(retrievers=[retriever_similarity, retriever_mmr, retriever_bm25])

    # remove redundant documents
    filter = EmbeddingsRedundantFilter(embeddings=embed_model)
    pipeline = DocumentCompressorPipeline(transformers=[filter])
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline, base_retriever=lotr
    )

    # rerank with Cohere
    compressor = CohereRerank(cohere_api_key=config['cohere_api_key'], top_n=top_n, model="rerank-multilingual-v3.0")
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=compression_retriever
    )

    return retriever