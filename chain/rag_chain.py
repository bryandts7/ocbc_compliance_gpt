from operator import itemgetter
from database.chat_store import MongoDBChatStore
from database.chat_store import RedisChatStore
from langchain_core.runnables import ConfigurableFieldSpec, RunnablePassthrough, RunnableLambda, RunnableBranch, RunnableParallel
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

from constant.sikepo.prompt import QA_SYSTEM_PROMPT_SIKEPO, ROUTER_PROMPT, REKAM_JEJAK_CONTEXT, KETENTUAN_ANSWERING_PROMPT
from constant.ojk.prompt import QA_SYSTEM_PROMPT_OJK
from constant.bi.prompt import QA_SYSTEM_PROMPT_BI
from constant.bi.prompt import QA_SYSTEM_PROMPT_BI
from constant.prompt import CONTEXTUALIZE_Q_PROMPT_STR, QA_SYSTEM_TEMPLATE

def retriever_to_list(results:list):
    return results[:]

def merge_context(results):
    merged = results["context_ojk"] + results["context_bi"] + results["context_sikepo"]
    return merged

def routing_ketentuan_chain(chain, llm_model):
    question_router = ketentuan_router_chain(llm_model, KETENTUAN_ANSWERING_PROMPT)
    result_chain =  RunnablePassthrough() | {
                        "question": itemgetter("question"), 
                        "answer": chain,
                        # "chat_history" : itemgetter("chat_history")
                     } | { 
                         "question": itemgetter("question"),
                         "answer": itemgetter("answer"),
                         "is_answered": question_router,
                        #  "chat_history" : itemgetter("chat_history")
                    } | RunnablePassthrough()
    return result_chain


def create_chain(retriever_ojk: BaseRetriever, retriever_sikepo_rekam: BaseRetriever, retriever_sikepo_ketentuan: BaseRetriever,
                 graph_chain:GraphCypherQAChain, llm_model: ModelName, retriever_bi: BaseRetriever = None):
    ojk_chain = create_ojk_chain(
        qa_system_prompt_str=QA_SYSTEM_PROMPT_OJK,
        llm_model=llm_model,
        retriever=retriever_ojk
    )
    bi_chain = create_bi_chain(
        qa_system_prompt_str=QA_SYSTEM_PROMPT_BI,
        llm_model=llm_model,
        retriever=retriever_bi
    )
    sikepo_ketentuan_chain = create_sikepo_ketentuan_chain(
        qa_system_prompt_str=QA_SYSTEM_PROMPT_SIKEPO,
        llm_model=llm_model,
        retriever=retriever_sikepo_ketentuan
    )
    sikepo_rekam_chain = create_sikepo_rekam_chain(
        # CONTEXTUALIZE_Q_PROMPT_SIKEPO, QA_SYSTEM_PROMPT_KETENTUAN_SIKEPO, REKAM_JEJAK_CONTEXT, retriever_sikepo_rekam, graph_chain, llm_model
        qa_system_prompt_str=QA_SYSTEM_PROMPT_SIKEPO,
        llm_model=llm_model,
        rekam_jejak_context=REKAM_JEJAK_CONTEXT,
        graph_chain=graph_chain,
        retriever=retriever_sikepo_rekam,
    )

    general_chain = PromptTemplate.from_template(
    """Answer 'Saya tidak tahu' """
    ) | llm_model | StrOutputParser()

    ketentuan_chain =   routing_ketentuan_chain(ojk_chain, llm_model) | RunnableBranch(
                            (lambda x: "YES" in x["is_answered"], itemgetter("answer") | RunnablePassthrough()),
                            (lambda x: "NO" in x["is_answered"], routing_ketentuan_chain(bi_chain, llm_model) | RunnableBranch(
                                (lambda x: "YES" in x["is_answered"], itemgetter("answer") | RunnablePassthrough()),
                                (lambda x: "NO" in x["is_answered"], routing_ketentuan_chain(sikepo_ketentuan_chain, llm_model) | itemgetter("answer") | RunnablePassthrough()),
                                general_chain,
                            )),
                            general_chain,
                            
                        )

    CONTEXTUALIZE_Q_PROMPT = PromptTemplate.from_template(CONTEXTUALIZE_Q_PROMPT_STR)
    _inputs_question = CONTEXTUALIZE_Q_PROMPT | llm_model | StrOutputParser()

    question_router = question_router_chain(llm_model, ROUTER_PROMPT)

    full_chain = _inputs_question | {
        "result": question_router | StrOutputParser(),
        "question": RunnablePassthrough(),
    } | RunnablePassthrough() | RunnableBranch(
        (lambda x: "rekam_jejak" in x["result"], sikepo_rekam_chain),
        (lambda x: "ketentuan_terkait" in x["result"], ketentuan_chain),
        general_chain,
    )

    return full_chain

def create_combined_chain(retriever_ojk: BaseRetriever, retriever_sikepo_rekam: BaseRetriever, retriever_sikepo_ketentuan: BaseRetriever,
                          graph_chain:GraphCypherQAChain, llm_model: ModelName, retriever_bi: BaseRetriever = None):
    
    CONTEXTUALIZE_Q_PROMPT = PromptTemplate.from_template(CONTEXTUALIZE_Q_PROMPT_STR)
    _inputs_question = CONTEXTUALIZE_Q_PROMPT | llm_model | StrOutputParser()

    QA_SYSTEM_PROMPT = PromptTemplate(input_variables=["context_ojk", "context_bi", "context_sikepo", "question"], template=QA_SYSTEM_TEMPLATE)

    general_chain = PromptTemplate.from_template(
    """Answer 'Saya tidak tahu' """
    ) | llm_model | StrOutputParser()

    sikepo_rekam_chain = create_sikepo_rekam_chain(
        # CONTEXTUALIZE_Q_PROMPT_SIKEPO, QA_SYSTEM_PROMPT_KETENTUAN_SIKEPO, REKAM_JEJAK_CONTEXT, retriever_sikepo_rekam, graph_chain, llm_model
        qa_system_prompt_str=QA_SYSTEM_PROMPT_SIKEPO,
        llm_model=llm_model,
        rekam_jejak_context=REKAM_JEJAK_CONTEXT,
        graph_chain=graph_chain,
        retriever=retriever_sikepo_rekam,
    )

    # _parallel = RunnableParallel(context_ojk = retriever_ojk | retriever_to_list, context_bi = retriever_bi | retriever_to_list, context_sikepo = retriever_sikepo_ketentuan | retriever_to_list, question = RunnablePassthrough())
    ketentuan_chain = RunnablePassthrough() | {"question": itemgetter("question")} | {
                        "context_ojk"       :   retriever_ojk | retriever_to_list,
                        "context_bi"        :   retriever_bi | retriever_to_list,
                        "context_sikepo"    :   retriever_sikepo_ketentuan | retriever_to_list,
                        "question"          :   itemgetter("question")
                        } | {
                        "rewrited question" :   itemgetter("question"),
                        "context"           :   RunnableLambda(merge_context),
                        "answer"            :   QA_SYSTEM_PROMPT | llm_model | StrOutputParser()
                        }

    question_router = question_router_chain(llm_model, ROUTER_PROMPT)

    full_chain = _inputs_question | {
        "result": question_router | StrOutputParser(),
        "question": RunnablePassthrough(),
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
