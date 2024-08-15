from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers import (
    ContextualCompressionRetriever,
    MergerRetriever,
)
from langchain_community.document_transformers import (
    EmbeddingsRedundantFilter, LongContextReorder
)
# from langchain.retrievers.document_compressors.flashrank_rerank import FlashrankRerank
# from langchain_community.document_compressors.rankllm_rerank import RankLLMRerank
from langchain.retrievers.document_compressors.base import DocumentCompressorPipeline
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda

from retriever.retriever_ojk.self_query_ojk import self_query_ojk
from retriever.retriever_sikepo.lotr_sikepo import to_documents

def get_retriever_ojk(vector_store: VectorStore, llm_model: BaseLanguageModel, embed_model: Embeddings, top_k: int = 8, config: dict = {}, with_self_query: bool = True):

    retriever_self_query_similarity = self_query_ojk(
        llm_model=llm_model,
        vector_store=vector_store,
        search_type="similarity",
    )

    retriever_self_query_mmr = self_query_ojk(
        llm_model=llm_model,
        vector_store=vector_store,
        search_type="mmr",
    )

    retriever_similarity = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={'k': top_k}
    )
    retriever_mmr = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={'k': top_k, 'lambda_mult': 0.85, 'fetch_k': 40}
    )

    lotr = retriever_mmr
    # merge retrievers
    if with_self_query:
        lotr = MergerRetriever(retrievers=[retriever_self_query_mmr, retriever_similarity])

    # remove redundant documents
    filter = EmbeddingsRedundantFilter(embeddings=embed_model)
    reordering = LongContextReorder()
    pipeline = DocumentCompressorPipeline(transformers=[filter, reordering])
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline, base_retriever=lotr
    )

    chain = compression_retriever | RunnableLambda(to_documents)

    # rerank with Cohere
    # compressor = CohereRerank(
    #     cohere_api_key=config['cohere_api_key'], top_n=top_n, model="rerank-multilingual-v3.0")
    # compressor = FlashrankRerank(top_n=top_n)
    # compressor = RankLLMRerank(top_n=top_n, model="gpt", gpt_model="gpt-3.5-turbo")
    
    # retriever = ContextualCompressionRetriever(
    #     base_compressor=compressor, base_retriever=compression_retriever
    # )

    return chain
