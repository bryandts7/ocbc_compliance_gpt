# ISI SEMUA PIPELINE CHAIN NYA, DARI INPUT ROUTING, PROMPT, SAMPAI OUTPUT CHAIN NYA

import json
from operator import itemgetter
from databases.chat_store import MongoDBChatStore
from databases.chat_store import RedisChatStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import (
    ConfigurableFieldSpec,
    RunnablePassthrough,
)
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables.history import RunnableWithMessageHistory
from utils.models import ModelName
from typing import Union


# ===== formatting functions =====
def _format_metadata(metadata):
    """Remove filename from metadata."""
    # check if file_name is in metadata
    if "file_name" in metadata:
        metadata.pop("file_name", None)
    return metadata


def _combine_documents(docs):
    """Combine documents into a single JSON string."""
    doc_list = [{"metadata": _format_metadata(
        doc.metadata), "page_content": doc.page_content} for doc in docs]
    return json.dumps(doc_list, indent=2)


# ===== INI MASIH OJK AJA, NTAR GABUNGING SEMUA LOGIC CHAIN NYA DISINI =====
def create_chain_with_chat_history(contextualize_q_prompt_str: str, qa_system_prompt_str: str, retriever: BaseRetriever, llm_model: ModelName, chat_store: Union[MongoDBChatStore, RedisChatStore]):
    CONTEXTUALIZE_Q_PROMPT_STR = contextualize_q_prompt_str
    QA_SYSTEM_PROMPT_STR = qa_system_prompt_str
    QA_PROMPT = ChatPromptTemplate.from_template(QA_SYSTEM_PROMPT_STR)
    CONTEXTUALIZE_Q_PROMPT = PromptTemplate.from_template(
        CONTEXTUALIZE_Q_PROMPT_STR)
    _inputs_question = CONTEXTUALIZE_Q_PROMPT | llm_model | StrOutputParser()
    _context_chain = _inputs_question | {
        "context": retriever | _combine_documents,
        "question": RunnablePassthrough()
    }
    conversational_qa_with_context_chain = (
        _context_chain
        | {
            "rewrited question": itemgetter("question"),
            "answer": QA_PROMPT | llm_model | StrOutputParser(),
            "context": itemgetter("context"),
        }
    )

    final_chain = RunnableWithMessageHistory(
        conversational_qa_with_context_chain,
        get_session_history=chat_store.get_session_history,
        input_messages_key="question",
        output_messages_key="answer",
        history_messages_key="chat_history",
        history_factory_config=[
            ConfigurableFieldSpec(
                id="user_id",
                annotation=str,
                name="User ID",
                description="Unique identifier for the user.",
                default="",
                is_shared=True,
            ),
            ConfigurableFieldSpec(
                id="conversation_id",
                annotation=str,
                name="Conversation ID",
                description="Unique identifier for the conversation.",
                default="",
                is_shared=True,
            ),
        ],
    )
    return final_chain


# ===== GET RESPONSE =====

def get_response(question: str, chain, user_id: str, conversation_id: str):
    response = chain.invoke(
        {"question": question},
        config={
            "configurable": {"user_id": user_id, "conversation_id": conversation_id}
        },
    )
    return response


def print_answer_stream(question: str, chain, user_id: str, conversation_id: str):
    for chunk in chain.stream({"question": question}, config={"configurable": {"user_id": user_id, "conversation_id": conversation_id}}):
        if 'answer' in chunk:
            print(chunk['answer'], end='', flush=True)
