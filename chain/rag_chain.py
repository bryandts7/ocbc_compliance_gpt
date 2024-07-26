# INI HANYA UNTUK OJK SAJA, NANTI BUAT LAGI DI rag_chain.py
# ISI SEMUA PIPELINE CHAIN NYA, DARI INPUT ROUTING, PROMPT, SAMPAI OUTPUT CHAIN NYA
import os
from operator import itemgetter
from databases.chat_store import MongoDBChatStore
from databases.chat_store import RedisChatStore
from langchain_core.runnables import ConfigurableFieldSpec, RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables.history import RunnableWithMessageHistory
from utils.models import ModelName
from typing import Union
from langchain_core.runnables.base import Runnable
from langchain.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


from chain.chain_routing import question_router_chain, ketentuan_router_chain
from chain.chain_ojk.chain_ojk import create_ojk_chain
from chain.chain_bi.chain_bi import create_bi_chain
from chain.chain_sikepo.chain_sikepo import create_sikepo_ketentuan_chain, create_sikepo_rekam_chain

from constant.sikepo.prompt import CONTEXTUALIZE_Q_PROMPT_SIKEPO, QA_SYSTEM_PROMPT_KETENTUAN_SIKEPO, ROUTER_PROMPT, REKAM_JEJAK_CONTEXT, KETENTUAN_ANSWERING_PROMPT
from constant.ojk.prompt import CONTEXTUALIZE_Q_PROMPT_OJK, QA_SYSTEM_PROMPT_OJK
from constant.bi.prompt import CONTEXTUALIZE_Q_PROMPT_BI, QA_SYSTEM_PROMPT_BI
from constant.bi.prompt import CONTEXTUALIZE_Q_PROMPT_BI, QA_SYSTEM_PROMPT_BI



def routing_ketentuan_chain(chain, llm_model):
    question_router = ketentuan_router_chain(llm_model, KETENTUAN_ANSWERING_PROMPT)
    result_chain =  RunnablePassthrough() | {
                        "question": itemgetter("question"), 
                        "answer": chain | itemgetter("answer"),
                        "chat_history" : itemgetter("chat_history")
                     } | { 
                         "question": itemgetter("question"),
                         "answer": itemgetter("answer"),
                         "is_answered": question_router,
                         "chat_history" : itemgetter("chat_history")}
                    # } | RunnablePassthrough()
    return result_chain


def create_chain(retriever_ojk: BaseRetriever, retriever_sikepo_rekam: BaseRetriever, retriever_sikepo_ketentuan: BaseRetriever,
                 graph_chain:GraphCypherQAChain, llm_model: ModelName, retriever_bi: BaseRetriever = None):
    ojk_chain = create_ojk_chain(CONTEXTUALIZE_Q_PROMPT_OJK, QA_SYSTEM_PROMPT_OJK, retriever_ojk, llm_model)
<<<<<<< HEAD
    bi_chain = create_bi_chain(CONTEXTUALIZE_Q_PROMPT_BI, QA_SYSTEM_PROMPT_BI, retriever_bi, llm_model)
=======
    bi_chain = create_ojk_chain(CONTEXTUALIZE_Q_PROMPT_BI, QA_SYSTEM_PROMPT_BI, retriever_bi, llm_model)
    bi_chain = create_bi_chain(CONTEXTUALIZE_Q_PROMPT_BI, QA_SYSTEM_PROMPT_BI, retriever_bi, llm_model)
    # bi_chain = create_ojk_chain(CONTEXTUALIZE_Q_PROMPT_OJK, QA_SYSTEM_PROMPT_OJK, retriever_ojk, llm_model)
>>>>>>> 60c3940860e8de1526588815a694eaa77d0df0ac
    sikepo_ketentuan_chain =create_sikepo_ketentuan_chain(CONTEXTUALIZE_Q_PROMPT_SIKEPO, QA_SYSTEM_PROMPT_KETENTUAN_SIKEPO, retriever_sikepo_ketentuan, llm_model)
    sikepo_rekam_chain = create_sikepo_rekam_chain(CONTEXTUALIZE_Q_PROMPT_SIKEPO, QA_SYSTEM_PROMPT_KETENTUAN_SIKEPO, REKAM_JEJAK_CONTEXT, retriever_sikepo_rekam, graph_chain, llm_model)

    question_router = question_router_chain(llm_model, ROUTER_PROMPT)
    general_chain = PromptTemplate.from_template(
    """Answer 'Saya tidak tahu' """
    ) | llm_model | StrOutputParser()

    ketentuan_chain =   routing_ketentuan_chain(ojk_chain, llm_model) | RunnableBranch(
                            (lambda x: "YES" in x["is_answered"], {"answer": itemgetter("answer")} | RunnablePassthrough()),
                            (lambda x: "NO" in x["is_answered"], routing_ketentuan_chain(bi_chain, llm_model) | RunnableBranch(
                                (lambda x: "YES" in x["is_answered"], {"answer": itemgetter("answer")} | RunnablePassthrough()),
                                (lambda x: "NO" in x["is_answered"], routing_ketentuan_chain(sikepo_ketentuan_chain, llm_model)),
                                general_chain,
                            )),
                            general_chain,
                            
                        )

    full_chain = {
        "result":question_router | StrOutputParser(),
        "question": itemgetter("question"),
        "chat_history" : itemgetter("chat_history")
    } | RunnablePassthrough() | RunnableBranch(
        (lambda x: "rekam_jejak" in x["result"], sikepo_rekam_chain),
        (lambda x: "ketentuan_terkait" in x["result"], ketentuan_chain),
        general_chain,
    )

    return full_chain

# ===== INI MASIH OJK AJA, NTAR GABUNGING SEMUA LOGIC CHAIN NYA DISINI =====
def create_chain_with_chat_history(chat_store: Union[MongoDBChatStore, RedisChatStore], final_chain:Runnable):
    final_chain = RunnableWithMessageHistory(
        final_chain,
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
