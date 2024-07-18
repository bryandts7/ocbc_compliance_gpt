# LOTR: Lord of the Retrievers
import os
from operator import itemgetter
from langchain.retrievers import (ContextualCompressionRetriever, MergerRetriever)
from langchain.retrievers.document_compressors.base import DocumentCompressorPipeline
from langchain_community.document_transformers import (EmbeddingsClusteringFilter,EmbeddingsRedundantFilter)
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.llms.cohere import Cohere
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain.memory.summary import SummarizerMixin
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from database.database import rekam_jejak_vector, ketentuan_terkait_vector
from database.graph_rag import graph_rag_chain
from handler.retriever.bm25 import bm25_ketentuan_retriever, bm25_rekam_retriever
from handler.retriever.self_query import self_retriever_ketentuan, self_retriever_rekam_jejak
from utils.azure_openai import azure_llm, azure_embeddings
from handler.routing import router_chain
from constants.prompt import RAG_FUSION_PROMPT, RAG_REKAM_JEJAK_PROMPT, SUMMARY_HISTORY_PROMPT


def history_summarize(history):
    global history_text
    sum_hist = history_sum.predict_new_summary(history, history_text)
    history_text = sum_hist
    os.write(1, f"History: {sum_hist}\n".encode())
    return sum_hist

def lotr_ketentuan(dic):
    contextualize_q_system_prompt  = f"""Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is. Write in the same language as the question (usually Indonesia).
    
    Question: {dic['question']}
    Chat History:
    {dic['history']}
    """

    qa_history_chain = llm | StrOutputParser()
    question_with_history = qa_history_chain.invoke(contextualize_q_system_prompt)
    print(question_with_history)
    try:
        lotr = MergerRetriever(retrievers=[ketentuan_terkait_self_retriever, ketentuan_terkait_retriever_mmr, ketentuan_terkait_retriever_sim])
        compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline, base_retriever=lotr)
        retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=compression_retriever)
        result = retriever.invoke(question_with_history)
        print("Self Retriever Success")
    except:
        lotr = MergerRetriever(retrievers=[ketentuan_terkait_retriever_bm25, ketentuan_terkait_retriever_mmr, ketentuan_terkait_retriever_sim])
        compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline, base_retriever=lotr)
        retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=compression_retriever)
        result = retriever.invoke(question_with_history)
    finally:
        print("Retrieved Context:", len(result))
        print(result)
        return result[:6]

def lotr_rekam(dic):
    contextualize_q_system_prompt  = f"""Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is. Write in the same language as the question (usually Indonesia).    
    
    Question: {dic['question']}
    Chat History:
    {dic['history']}
    """

    qa_history_chain = llm | StrOutputParser()
    question_with_history = qa_history_chain.invoke(contextualize_q_system_prompt)
    print(question_with_history)

    try:
        lotr = MergerRetriever(retrievers=[rekam_jejak_self_retriever, rekam_jejak_retriever_mmr, rekam_jejak_retriever_sim])
        compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline, base_retriever=lotr)
        retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=compression_retriever)
        result = retriever.invoke(question_with_history)
        print("Self Retriever Success")
    except:
        lotr = MergerRetriever(retrievers=[rekam_jejak_retriever_bm25, rekam_jejak_retriever_mmr, rekam_jejak_retriever_sim])
        compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline, base_retriever=lotr)
        retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=compression_retriever)
        result = retriever.invoke(question_with_history)
    finally:
        print("Retrieved Context:", len(result))
        print(result)
        return result[:6]


def choose_retriever(result):
    os.write(1, f"\nRetriever routed to: {result['result']}\n".encode())
    if result["result"] == "rekam_jejak":
        parallel = RunnableParallel(unstructured=RunnableLambda(lotr_rekam), structured=rekam_jejak_graph_chain)
        chain = {"question": itemgetter("question"), "query": itemgetter("question"),
                 "history" : itemgetter("history") | RunnableLambda(history_summarize)} | parallel | context_rekam_jejak
        return chain
    elif result["result"] == "ketentuan_terkait":
        chain = {"question": itemgetter("question"),
                 "history" : itemgetter("history") | RunnableLambda(history_summarize)} | RunnableLambda(lotr_ketentuan)
        return chain

def lotr_context_chain():
    context_chain = {
    "result":router,
    "question": itemgetter("question"),
    "history" : itemgetter("history")
    } | RunnablePassthrough() | {"context": RunnableLambda(choose_retriever), "question": itemgetter("question"), "history" : itemgetter("history") }

    return context_chain  
    
SUMMARY_PROMPT = PromptTemplate(
    input_variables=["summary", "new_lines"], template=SUMMARY_HISTORY_PROMPT
)
llm = azure_llm()
embedding = azure_embeddings()
compressor = CohereRerank(model="rerank-multilingual-v3.0", top_n=10)
router = router_chain()

filter = EmbeddingsRedundantFilter(embeddings=embedding)
filter_ordered_by_retriever = EmbeddingsClusteringFilter(
    embeddings=embedding,
    num_clusters=10,
    num_closest=1,
    sorted=True,
)
pipeline = DocumentCompressorPipeline(transformers=[filter_ordered_by_retriever, filter])

history_sum = SummarizerMixin(llm=llm, prompt=SUMMARY_PROMPT)
history_text = "-"

rekam_jejak_retriever_mmr = rekam_jejak_vector().as_retriever(search_type="mmr", k=5)
rekam_jejak_retriever_sim = rekam_jejak_vector().as_retriever(search_type="similarity", k=5)
rekam_jejak_retriever_bm25 = bm25_rekam_retriever()
rekam_jejak_retriever_bm25.k = 10
rekam_jejak_self_retriever = self_retriever_rekam_jejak()
rekam_jejak_graph_chain = graph_rag_chain()

ketentuan_terkait_retriever_mmr = ketentuan_terkait_vector().as_retriever(search_type="mmr")
ketentuan_terkait_retriever_sim = ketentuan_terkait_vector().as_retriever(search_type="similarity")
ketentuan_terkait_retriever_bm25 = bm25_ketentuan_retriever()
ketentuan_terkait_retriever_bm25.k = 10
ketentuan_terkait_self_retriever = self_retriever_ketentuan()

template_rekam_jejak = RAG_REKAM_JEJAK_PROMPT
context_rekam_jejak = PromptTemplate(
    input_variables=["unstructured", "structured"], template=template_rekam_jejak
)
