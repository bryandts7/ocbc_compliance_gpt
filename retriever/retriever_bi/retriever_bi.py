from langchain.retrievers.document_compressors.base import DocumentCompressorPipeline
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from langchain.retrievers import EnsembleRetriever

def get_retriever_bi(vector_store: VectorStore, llm_model: BaseLanguageModel, embed_model: Embeddings, top_n: int = 7, top_k:int = 20, config: dict = {}, metadata_field_info: list = []):
    retriever = vector_store.as_retriever(search_kwargs={"k": 30})

    return retriever