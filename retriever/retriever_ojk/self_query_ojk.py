from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)
from langchain_core.language_models.base import BaseLanguageModel
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.vectorstores import VectorStore


# Define metadata field information
metadata_field_info = [
    AttributeInfo(
        name="title",
        description="The title of the document of regulation",
        type="string",
    ),
    AttributeInfo(
        name="sector",
        description="""The sector of the regulation""",
        type="string",
    ),
    AttributeInfo(
        name="subsector",
        description="The subsector of the regulation",
        type="string",
    ),
    AttributeInfo(
        name="regulation_type",
        description="""The type of the regulation""",
        type="string",
    ),
    AttributeInfo(
        name="regulation_number",
        description="The number of the regulation",
        type="string",
    ),
    AttributeInfo(
        name="effective_date",
        description="The effective date of the regulation in format DD Month YYYY, e.g. 1 Januari 2021",
        type="string",
    ),
]

# Define document content description
document_content_description = "The content of the document"

# Define prompt
schema_prompt = """
Please provide the schema of the structured query. Only the following attributes are allowed:
- title
- sector
- subsector
- regulation_type
- regulation_number
- effective_date

Ensure that user queries are interpreted correctly by mapping common phrases to the corresponding attributes:
- "judul" or any mention of a title should be interpreted as the attribute 'title'
- "sektor" or any mention of a sector should be interpreted as the attribute 'sector'
- "subsektor" or any mention of a subsector should be interpreted as the attribute 'subsector'
- "tipe regulasi" or any mention of a regulation type should be interpreted as the attribute 'regulation_type'
- "nomor regulasi" or any mention of a regulation number should be interpreted as the attribute 'regulation_number'
- "tanggal berlaku" or any mention of an effective date should be interpreted as the attribute 'effective_date'
"""

# Create query constructor
def self_query_ojk(llm_model: BaseLanguageModel, vector_store: VectorStore, search_type: str = "similarity") -> SelfQueryRetriever:
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


    