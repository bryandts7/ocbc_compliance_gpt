from operator import itemgetter
from database.chat_store import ElasticChatStore
from langchain_core.runnables import ConfigurableFieldSpec, RunnablePassthrough, RunnableLambda, RunnableBranch, RunnableParallel
from langchain_core.retrievers import BaseRetriever
from utils.models import ModelName
import json
from langchain_core.runnables.base import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory
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
from constant.prompt import CONTEXTUALIZE_Q_PROMPT_STR, QA_SYSTEM_TEMPLATE, QA_SYSTEM_TEMPLATE_COMBINED_ANSWER


def retriever_to_list(results: list):
    print(results)
    return str(results)


def merge_context(results):
    merged = results["context_ojk"] + \
        results["context_bi"] + results["context_sikepo"]
    return merged

def merge_answer(results):
    merged = [results["answer_ojk"], results["answer_bi"], results["answer_sikepo"]]
    return merged

def routing_ketentuan_chain(chain, llm_model):
    question_router = ketentuan_router_chain(
        llm_model, KETENTUAN_ANSWERING_PROMPT)
    result_chain = RunnablePassthrough() | {
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


def create_sequential_chain(retriever_ojk: BaseRetriever, retriever_sikepo_rekam: BaseRetriever, retriever_sikepo_ketentuan: BaseRetriever,
                            graph_chain: GraphCypherQAChain, llm_model: ModelName, retriever_bi: BaseRetriever = None):
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

    ketentuan_chain = routing_ketentuan_chain(ojk_chain, llm_model) | RunnableBranch(
        (lambda x: "YES" in x["is_answered"],
         itemgetter("answer") | RunnablePassthrough()),
        (lambda x: "NO" in x["is_answered"], routing_ketentuan_chain(bi_chain, llm_model) | RunnableBranch(
            (lambda x: "YES" in x["is_answered"], itemgetter(
                "answer") | RunnablePassthrough()),
            (lambda x: "NO" in x["is_answered"], routing_ketentuan_chain(
                sikepo_ketentuan_chain, llm_model) | itemgetter("answer") | RunnablePassthrough()),
            general_chain,
        )),
        general_chain,

    )

    CONTEXTUALIZE_Q_PROMPT = PromptTemplate.from_template(
        CONTEXTUALIZE_Q_PROMPT_STR)
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


def printing(results):
    print(results)

# Still error


def create_combined_answer_chain(retriever_ojk: BaseRetriever, retriever_sikepo_rekam: BaseRetriever, retriever_sikepo_ketentuan: BaseRetriever,
                                 graph_chain: GraphCypherQAChain, llm_model: ModelName, retriever_bi: BaseRetriever = None, best_llm = None, efficient_llm = None):
    if best_llm is None:
        best_llm = llm_model
    if efficient_llm is None:
        efficient_llm = llm_model

    CONTEXTUALIZE_Q_PROMPT = PromptTemplate.from_template(
        CONTEXTUALIZE_Q_PROMPT_STR)
    _inputs_question = CONTEXTUALIZE_Q_PROMPT | efficient_llm | StrOutputParser()

    question_router = question_router_chain(best_llm, ROUTER_PROMPT)
    QA_SYSTEM_PROMPT = PromptTemplate(input_variables=[
                                      "answer_ojk", "answer_bi", "answer_sikepo", "question"], template=QA_SYSTEM_TEMPLATE_COMBINED_ANSWER)

    ojk_chain = create_ojk_chain(
        qa_system_prompt_str=QA_SYSTEM_PROMPT_OJK,
        llm_model=llm_model,
        retriever=retriever_ojk
    )
    bi_chain = create_bi_chain(
        qa_system_prompt_str=QA_SYSTEM_PROMPT_BI,
        llm_model=efficient_llm,
        retriever=retriever_bi
    )
    sikepo_ketentuan_chain = create_sikepo_ketentuan_chain(
        qa_system_prompt_str=QA_SYSTEM_PROMPT_SIKEPO,
        llm_model=efficient_llm,
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

    ketentuan_chain = RunnablePassthrough() | {"question": itemgetter("question")} | {
        "chain_ojk":   ojk_chain,
        "chain_bi":   bi_chain,
        "chain_sikepo":   sikepo_ketentuan_chain,
        "question":   itemgetter("question")
    } | {
        "answer_ojk":   itemgetter("chain_ojk") | RunnablePassthrough() | itemgetter("answer") | RunnablePassthrough(),
        "context_ojk":   itemgetter("chain_ojk") | RunnablePassthrough() | itemgetter("context") | RunnablePassthrough(),
        "answer_bi":   itemgetter("chain_bi") | RunnablePassthrough() | itemgetter("answer") | RunnablePassthrough(),
        "context_bi":   itemgetter("chain_bi") | RunnablePassthrough() | itemgetter("context") | RunnablePassthrough(),
        "answer_sikepo":   itemgetter("chain_sikepo") | RunnablePassthrough() | itemgetter("answer") | RunnablePassthrough(),
        "context_sikepo":   itemgetter("chain_sikepo") | RunnablePassthrough() | itemgetter("context") | RunnablePassthrough(),
        "question":   itemgetter("question")
    } | {
        "rewrited question":   itemgetter("question"),
        "context":   RunnableLambda(merge_context),
        "context_text": RunnableLambda(merge_answer),
        "answer":   QA_SYSTEM_PROMPT | best_llm | StrOutputParser()
    }

    full_chain = _inputs_question | {
        "result": question_router | StrOutputParser(),
        "question": RunnablePassthrough(),
    } | RunnablePassthrough() | RunnableBranch(
        (lambda x: "rekam_jejak" in x["result"], sikepo_rekam_chain),
        (lambda x: "ketentuan_terkait" in x["result"], ketentuan_chain),
        general_chain,
    )

    return full_chain


def create_combined_context_chain(retriever_ojk: BaseRetriever, retriever_sikepo_rekam: BaseRetriever, retriever_sikepo_ketentuan: BaseRetriever,
                                  graph_chain: GraphCypherQAChain, llm_model: ModelName, retriever_bi: BaseRetriever = None, best_llm = None, efficient_llm = None):
    if best_llm is None:
        best_llm = llm_model
    if efficient_llm is None:
        efficient_llm = llm_model

    CONTEXTUALIZE_Q_PROMPT = PromptTemplate.from_template(
        CONTEXTUALIZE_Q_PROMPT_STR)
    _inputs_question = CONTEXTUALIZE_Q_PROMPT | efficient_llm | StrOutputParser()

    QA_SYSTEM_PROMPT = PromptTemplate(input_variables=[
                                      "context_ojk", "context_bi", "context_sikepo", "question"], template=QA_SYSTEM_TEMPLATE)

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
    _parallel = RunnableParallel(context_ojk=retriever_ojk, context_bi=retriever_bi,
                                 context_sikepo=retriever_sikepo_ketentuan, question=RunnablePassthrough())

    ketentuan_chain = RunnablePassthrough() | itemgetter("question") | _parallel | {
        "rewrited question":   itemgetter("question"),
        "context":   RunnableLambda(merge_context),
        "answer":   QA_SYSTEM_PROMPT | llm_model | StrOutputParser()
    }

    question_router = question_router_chain(best_llm, ROUTER_PROMPT)

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
def create_chain_with_chat_history(chat_store: ElasticChatStore, final_chain: Runnable):
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


async def print_answer_stream(question: str, chain: RunnableWithMessageHistory, user_id: str, conversation_id: str):
    async for chunk in chain.astream({"question": question}, config={"configurable": {"user_id": user_id, "conversation_id": conversation_id}}):
        if 'answer' in chunk:
            # yield f"data: {chunk['answer'] or ''}\n\n"
            json_chunk = json.dumps({"answer": chunk['answer'] or ''})
            yield f"data: {json_chunk}\n\n"
            # yield f"data: {json_chunk}\n\n"
            


async def print_answer_stream2(question: str, chain: RunnableWithMessageHistory, user_id: str, conversation_id: str):
    async for chunk in chain.astream_events({"question": question}, config={"configurable": {"user_id": user_id, "conversation_id": conversation_id}}, version="v1"):
        # if 'answer' in chunk:
        #     print(chunk['answer'], end='', flush=True)
        print(chunk, end='\n', flush=True)
