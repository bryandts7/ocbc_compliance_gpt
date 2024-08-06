# ISI RETRIEVER DARI SIKEPO KETENTUAN TERKAIT, OUTPUT HARUS SATU RETRIEVER SAJA
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers import (
    ContextualCompressionRetriever,
    MergerRetriever,
)
from langchain_community.document_transformers import (
    EmbeddingsRedundantFilter, EmbeddingsClusteringFilter
)
from langchain.retrievers.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_community.document_compressors.rankllm_rerank import RankLLMRerank
from langchain.retrievers.document_compressors.base import DocumentCompressorPipeline
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.embeddings import Embeddings

from retriever.retriever_sikepo.self_query_sikepo import self_query_retriever_sikepo

# ini panggil sesuai vector store saja, jadi gak perlu defin 2 kali
# misal : lotr_ketentuan_terkait = lotr_siekepo(vector_store=vector_store_ketentuan_terkait, ...)
# untuk chain di folder terpisah soalnya responsibility nya beda


def lotr_sikepo(vector_store: VectorStore, llm_model: BaseLanguageModel, embed_model: Embeddings, config: dict = {}, top_n: int = 10):

    retriever_mmr = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.25})
    retriever_similarity = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 6})
    self_query_retriever = self_query_retriever_sikepo(
        llm_model=llm_model, vector_store=vector_store)
    # bm25_retriever = bm25_retriever_sikepo() # ini fungsinya belom di define yee, jadi gak bisa dipanggil, define di `retriver/retriever_sikepo/bm25_retriever_sikepo.py

    # try:
    lotr = MergerRetriever(
        retrievers=[self_query_retriever, retriever_mmr, retriever_similarity])
    # except:
    #     lotr = MergerRetriever(retrievers=[retriever_mmr, retriever_similarity, bm25_retriever])

    # remove redundant documents
    filter = EmbeddingsRedundantFilter(embeddings=embed_model)
    pipeline = DocumentCompressorPipeline(transformers=[filter])
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline, base_retriever=lotr
    )

    # rerank with Cohere
    # compressor = CohereRerank(
    #     cohere_api_key=config['cohere_api_key'], top_n=top_n, model="rerank-multilingual-v3.0")
    # compressor = FlashrankRerank(top_n=top_n)
    compressor = RankLLMRerank(top_n=top_n, model="gpt", gpt_model="gpt-3.5-turbo")
    
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=compression_retriever
    )

    return retriever
