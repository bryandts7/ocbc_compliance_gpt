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


metadata_field_info = [
    AttributeInfo(
        name="title",
        description="The title of the document of regulation",
        type="string",
    ),
    AttributeInfo(
        name="sector",
        description="The sector of the regulation",
        type="string",
    ),
    AttributeInfo(
        name="subsector",
        description="The subsector of the regulation",
        type="string",
    ),
    AttributeInfo(
        name="regulation_type",
        description="The type of the regulation",
        type="string",
    ),
    AttributeInfo(
        name="regulation_number",
        description="The number of the regulation",
        type="string",
    ),
    AttributeInfo(
        name="effective_date",
        description="The effective date of the regulation",
        type="string",
    ),
]

document_content_description = "The content of the document"


def get_retriever_ojk(vector_store: VectorStore, llm_model: BaseLanguageModel, embed_model: Embeddings, top_n: int = 7, top_k:int = 20, config: dict = {}):

    top_k = top_k // 4

    retriever_self_query_similarity = SelfQueryRetriever.from_llm(
    llm=llm_model,
    vectorstore=vector_store,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
    enable_limit=True,
    search_type="similarity",
    search_kwargs={"k": top_k},
    )
    
    retriever_self_query_mmr = SelfQueryRetriever.from_llm(
    llm=llm_model,
    vectorstore=vector_store,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
    enable_limit=True,
    search_type="mmr",
    search_kwargs={"k": top_k},
    )

    retriever_similarity = vector_store.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": top_k}
    )
    retriever_mmr = vector_store.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": top_k}
    )

    # merge retrievers
    lotr = MergerRetriever(retrievers=[retriever_similarity, retriever_mmr, retriever_self_query_mmr, retriever_self_query_similarity])

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