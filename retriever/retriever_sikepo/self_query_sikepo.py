from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.vectorstores import VectorStore

from retriever.self_query import self_query


metadata_field_info = [
    AttributeInfo(
        name="Ketentuan",
        description="Pasal spesifik dalam peraturan (Specific Chapter/Article in the regulation)",
        type="string",
    ),
    AttributeInfo(
        name="Nomor Ketentuan",
        description="Nomor dari ketentuan (Number of regulation)",
        type="string",
    ),
    AttributeInfo(
        name="Tanggal Ketentuan",
        description="Tanggal ketika ketentuan diterbitkan",
        type="string",
    ),
]

document_content_description = "Isi Ketentuan dari Peraturan"

# Create query constructor


def self_query_retriever_sikepo(llm_model: BaseLanguageModel, vector_store: VectorStore, search_type: str = "mmr", top_k: int = 8) -> SelfQueryRetriever:
    return self_query(llm_model, vector_store, document_content_description, metadata_field_info, search_type=search_type, top_k=top_k)
