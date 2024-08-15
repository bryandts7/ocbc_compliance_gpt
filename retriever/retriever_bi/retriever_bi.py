from langchain.retrievers import (ContextualCompressionRetriever,
                                  MergerRetriever)
from langchain.retrievers.document_compressors.base import \
    DocumentCompressorPipeline
from langchain_cohere import CohereRerank
# from langchain.retrievers.document_compressors.flashrank_rerank import FlashrankRerank
# from langchain_community.document_compressors.rankllm_rerank import RankLLMRerank
from langchain_community.document_transformers import EmbeddingsRedundantFilter, LongContextReorder
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.vectorstores import VectorStore
from langchain_core.runnables import RunnableLambda
from retriever.retriever_sikepo.lotr_sikepo import to_documents


# all_documents_file = gzip.open(f'retriever/retriever_bi/all_documents.pkl.gz','rb')
# all_documents = pickle.load(all_documents_file)


def get_retriever_bi(vector_store: VectorStore, llm_model: BaseLanguageModel, embed_model: Embeddings, top_k: int = 8, config: dict = {}):

    retriever_similarity = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={'k': top_k}
    )
    retriever_mmr = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={'k': top_k, 'lambda_mult': 0.85}
    )

    # merge retrievers
    lotr = retriever_mmr

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
