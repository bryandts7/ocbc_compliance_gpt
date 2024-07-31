from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)
from langchain_core.language_models.base import BaseLanguageModel
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.vectorstores import VectorStore
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union
from langchain_core.prompts import BasePromptTemplate

from constant.prompt import DEFAULT_SCHEMA_PROMPT


def self_query(llm_model: BaseLanguageModel, vector_store: VectorStore, document_content_description:str, metadata_field_info: Sequence[Union[AttributeInfo, dict]], search_type: str = "similarity", schema_prompt: BasePromptTemplate = DEFAULT_SCHEMA_PROMPT  ) -> SelfQueryRetriever:
    prompt = get_query_constructor_prompt(
        document_contents=document_content_description,
        attribute_info=metadata_field_info,
        schema_prompt=schema_prompt,
    )
    output_parser = StructuredQueryOutputParser.from_components()
    query_constructor = prompt | llm_model | output_parser

    retriever = SelfQueryRetriever(
        query_constructor=query_constructor,
        vectorstore=vector_store,
        search_type=search_type,
    )

    return retriever