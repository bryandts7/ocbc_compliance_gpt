from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.vectorstores import VectorStore
from langchain.retrievers.self_query.pgvector import PGVectorTranslator


metadata_field_info = [
    AttributeInfo(
        name="Jenis Ketentuan",
        description="Jenis peraturan atau ketentuan",
        type="string",
    ),
    AttributeInfo(
        name="Judul Ketentuan",
        description="Judul peraturan atau ketentuan",
        type="string",
    ),
    AttributeInfo(
        name="Ketentuan",
        description="Pasal atau ketentuan spesifik dalam peraturan",
        type="string",
    ),
    AttributeInfo(
        name="Kodifikasi Ketentuan",
        description="Kategori kodifikasi ketentuan",
        type="string",
    ),
    AttributeInfo(
        name="Nomor Ketentuan",
        description="Nomor dari ketentuan",
        type="string",
    ),
    AttributeInfo(
        name="Referensi",
        description="Referensi terkait ketentuan",
        type="string",
    ),
    AttributeInfo(
        name="Tanggal Ketentuan",
        description="Tanggal ketika ketentuan diterbitkan",
        type="string",
    ),
]

document_content_description = "Isi Ketentuan dari Peraturan"

from typing import Dict, Tuple, Union
from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)
from langchain.retrievers.self_query.pgvector import PGVectorTranslator

class CustomPGVectorTranslator(PGVectorTranslator):
    """Custom PGVectorTranslator to add `$` prefix to comparators."""

    def _format_func(self, func: Union[Operator, Comparator]) -> str:
        self._validate_func(func)
        return f"${func.value}"  # Add the `$` prefix

# ini juga gua buat satu aja jadi kalau mau make self self_query_retriever_ketentuan_terkait, tinggal pake vector_store ketentuan terkait aja
def self_query_retriever_sikepo(llm_model: BaseLanguageModel, vector_store: VectorStore, with_limit: bool = False):
    retriever = SelfQueryRetriever.from_llm(
        llm=llm_model,
        vectorstore=vector_store,
        document_contents=document_content_description,
        metadata_field_info=metadata_field_info,
        verbose=True,
        enable_limit=with_limit,
    )
    return retriever