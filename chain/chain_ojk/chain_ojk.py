# INI HANYA UNTUK OJK SAJA, NANTI BUAT LAGI DI rag_chain.py
# ISI SEMUA PIPELINE CHAIN NYA, DARI INPUT ROUTING, PROMPT, SAMPAI OUTPUT CHAIN NYA

# import json
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.retrievers import BaseRetriever
from utils.models import ModelName

# ===== formatting functions =====
# def _format_metadata(metadata):
#     """Remove filename from metadata."""
#     # check if file_name is in metadata, if so remove it
#     if "file_name" in metadata:
#         metadata.pop("file_name", None)
#     return metadata


# def _combine_documents(docs):
#     """Combine documents into a single JSON string."""
#     doc_list = [{"metadata": _format_metadata(
#         doc.metadata), "page_content": doc.page_content} for doc in docs]
#     return json.dumps(doc_list, indent=2)


# ===== INI MASIH OJK AJA, NTAR GABUNGING SEMUA LOGIC CHAIN NYA DISINI =====
def create_ojk_chain(qa_system_prompt_str: str, retriever: BaseRetriever, llm_model: ModelName):
    _context_chain = RunnablePassthrough() | itemgetter("question") | {
        # "context": retriever | _combine_documents,
        "context": retriever,
        "question": RunnablePassthrough()
    }
    QA_SYSTEM_PROMPT_STR = qa_system_prompt_str
    QA_PROMPT = ChatPromptTemplate.from_template(QA_SYSTEM_PROMPT_STR)
    conversational_qa_with_context_chain = (
        _context_chain
        | {
            "rewrited question": itemgetter("question"),
            "answer": QA_PROMPT | llm_model | StrOutputParser(),
            "context": itemgetter("context"),
        } | RunnablePassthrough()
    )
    return conversational_qa_with_context_chain
