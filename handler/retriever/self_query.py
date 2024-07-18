from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

from database.database import rekam_jejak_vector, ketentuan_terkait_vector
from utils.azure_openai import azure_llm

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

llm = azure_llm()

def self_retriever_rekam_jejak():
    rekam_jejak = rekam_jejak_vector()
    retriever = SelfQueryRetriever.from_llm(
        llm, rekam_jejak, document_content_description, metadata_field_info, verbose=True
    )
    return retriever

def self_retriever_ketentuan():
    ketentuan_terkait = ketentuan_terkait_vector()
    retriever = SelfQueryRetriever.from_llm(
        llm, ketentuan_terkait, document_content_description, metadata_field_info, verbose=True
    )
    return retriever

