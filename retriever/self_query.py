from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)
from langchain_core.language_models.base import BaseLanguageModel
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.vectorstores import VectorStore
from typing import Sequence, Union
from langchain_core.prompts import BasePromptTemplate
from langchain_community.query_constructors.elasticsearch import ElasticsearchTranslator

from constant.prompt import DEFAULT_SCHEMA_PROMPT


def self_query(llm_model: BaseLanguageModel, vector_store: VectorStore, document_content_description: str, metadata_field_info: Sequence[Union[AttributeInfo, dict]], search_type: str = "similarity", schema_prompt: BasePromptTemplate = DEFAULT_SCHEMA_PROMPT, top_k: int = 8) -> SelfQueryRetriever:
    es_translator = ElasticsearchTranslator()
    prompt = get_query_constructor_prompt(
        document_contents=document_content_description,
        attribute_info=metadata_field_info,
        schema_prompt=schema_prompt,
        allowed_comparators=es_translator.allowed_comparators,
        allowed_operators=es_translator.allowed_operators
    )

    output_parser = StructuredQueryOutputParser.from_components()
    query_constructor = prompt | llm_model | output_parser

    retriever = SelfQueryRetriever(
        query_constructor=query_constructor,
        vectorstore=vector_store,
        search_type=search_type,
        search_kwargs={'k': top_k}
    )

    if search_type == "mmr":
        retriever = SelfQueryRetriever(
        query_constructor=query_constructor,
        vectorstore=vector_store,
        search_type=search_type,
        search_kwargs={'k': top_k, 'lambda_mult': 0.75, 'fetch_k': 40}
    )

    return retriever
