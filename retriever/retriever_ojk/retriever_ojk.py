from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers import (
    ContextualCompressionRetriever,
    MergerRetriever,
)
from langchain_community.document_transformers import (
    EmbeddingsRedundantFilter,
)
from langchain.retrievers.document_compressors.base import DocumentCompressorPipeline
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.embeddings import Embeddings

from retriever.retriever_ojk.self_query_ojk import self_query_ojk


def get_retriever_ojk(vector_store: VectorStore, llm_model: BaseLanguageModel, embed_model: Embeddings, top_n: int = 7, config: dict = {}):

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
    )
    retriever_mmr = vector_store.as_retriever(
        search_type="mmr",
    )

    # merge retrievers
    lotr = MergerRetriever(retrievers=[retriever_self_query_similarity,
                           retriever_self_query_mmr, retriever_similarity, retriever_mmr])
    # lotr = MergerRetriever(retrievers=[retriever_similarity, retriever_mmr])

    # remove redundant documents
    filter = EmbeddingsRedundantFilter(embeddings=embed_model)
    pipeline = DocumentCompressorPipeline(transformers=[filter])
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline, base_retriever=lotr
    )

    # rerank with Cohere
    compressor = CohereRerank(
        cohere_api_key=config['cohere_api_key'], top_n=top_n, model="rerank-multilingual-v3.0")
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=compression_retriever
    )

    return retriever
