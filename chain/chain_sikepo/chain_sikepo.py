from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.retrievers import BaseRetriever
from utils.models import ModelName
from langchain.chains.graph_qa.cypher import GraphCypherQAChain


def create_sikepo_ketentuan_chain(qa_system_prompt_str: str, retriever: BaseRetriever, llm_model: ModelName):
    _context_chain = RunnablePassthrough() | itemgetter("question") | {
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


def create_sikepo_rekam_chain(qa_system_prompt_str: str, rekam_jejak_context: str, retriever: BaseRetriever, graph_chain: GraphCypherQAChain, llm_model: ModelName):
    QA_SYSTEM_PROMPT_REKAM_STR = qa_system_prompt_str
    REKAM_JEJAK_CONTEXT = rekam_jejak_context
    QA_PROMPT = ChatPromptTemplate.from_template(QA_SYSTEM_PROMPT_REKAM_STR)
    CONTEXT_PROMPT = PromptTemplate(
        input_variables=["unstructured", "structured"], template=REKAM_JEJAK_CONTEXT)

    _parallel_runnable = RunnableParallel(structured=graph_chain, unstructured=itemgetter(
        "question") | RunnablePassthrough() | retriever)
    _context_chain = RunnablePassthrough() | itemgetter("question") | {
        "question": RunnablePassthrough(),
        "query": RunnablePassthrough()
    } | {
        "context": _parallel_runnable | CONTEXT_PROMPT,
        "question": itemgetter("question")
    }
    conversational_qa_with_context_chain = (
        _context_chain
        | {
            "rewrited question": itemgetter("question"),
            "answer": QA_PROMPT | llm_model | StrOutputParser(),
            "context": itemgetter("context"),
        } | RunnablePassthrough()
    )
    return conversational_qa_with_context_chain
