import os
from operator import itemgetter
from bson import ObjectId
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain.memory.summary import SummarizerMixin

from database.database import rekam_jejak_vector, ketentuan_terkait_vector
from utils.azure_openai import azure_llm
from handler.routing import router_chain
from constants.prompt import RAG_FUSION_PROMPT, RAG_REKAM_JEJAK_PROMPT, SUMMARY_HISTORY_PROMPT
from database.graph_rag import graph_rag_chain
from handler.retriever.bm25 import bm25_ketentuan_retriever, bm25_rekam_retriever
from handler.retriever.self_query import self_retriever_ketentuan, self_retriever_rekam_jejak

# Convert ObjectId to string before serialization
def convert_objectid_to_string(doc):
    if '_id' in doc.metadata:
        doc.metadata['_id'] = str(doc.metadata['_id'])
    return doc

# Convert string back to ObjectId after deserialization
def convert_string_to_objectid(doc):
    if '_id' in doc.metadata:
        doc.metadata['_id'] = ObjectId(doc.metadata['_id'])
        del doc.metadata['embedding']
    return doc

def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(convert_objectid_to_string(doc))
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (convert_string_to_objectid(loads(doc)), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    os.write(1, f"Retrieved Documents: {len(reranked_results)}\n".encode())

    #Top 8 Result
    best_result = [reranked_results[i][0] for i in range(min(8, len(reranked_results)))]

    # Return the best 8 results according to fused score as a list of document
    return best_result

def history_summarize(history):
    global history_text
    sum_hist = history_sum.predict_new_summary(history, history_text)
    history_text = sum_hist
    os.write(1, f"History: {sum_hist}\n".encode())
    return sum_hist

def multi_retrievers_rekam(queries):
    """ Runnable Function forwarding from 3 generated queries for Fusion with 3 different retrievers.
        Try Pre-Filtering Metadata through Self-Query first. If error, then using BM25.  """
    try:
        pre_filter = rekam_jejak_self_retriever.invoke(queries[0])
        os.write(1, f"Using Self-Retriever\n".encode())
        res = [pre_filter, pre_filter, rekam_jejak_retriever_mmr.invoke(queries[1]), rekam_jejak_retriever_sim.invoke(queries[2])]
    except:
        pre_filter = rekam_jejak_retriever_bm25.invoke(queries[0])
        os.write(1, f"Using BM25\n".encode())
        res = [pre_filter, rekam_jejak_retriever_mmr.invoke(queries[1]), rekam_jejak_retriever_sim.invoke(queries[2])]
    finally:
        return res

def multi_retrievers_ketentuan(queries):
    """ Runnable Function forwarding from 3 generated queries for Fusion with 3 different retrievers. 
        Try Pre-Filtering Metadata through Self-Query first. If error, then using BM25. """
    try:
        pre_filter = ketentuan_terkait_self_retriever.invoke(queries[0])
        os.write(1, f"Using Self-Retriever\n".encode())
        res = [pre_filter, pre_filter, ketentuan_terkait_retriever_mmr.invoke(queries[1]), ketentuan_terkait_retriever_sim.invoke(queries[2])]
    except:
        pre_filter = ketentuan_terkait_retriever_bm25.invoke(queries[0])
        os.write(1, f"Using BM25\n".encode())
        res = [pre_filter, ketentuan_terkait_retriever_mmr.invoke(queries[1]), ketentuan_terkait_retriever_sim.invoke(queries[2])]
    finally:
        return res

def choose_retriever(result):
    os.write(1, f"\nRetriever routed to: {result['result']}\n".encode())
    if result["result"] == "rekam_jejak":
        parallel = RunnableParallel(unstructured=rekam_jejak_chain_rag_fusion, structured=rekam_jejak_graph_chain)
        chain = {"question": itemgetter("question"), "query": itemgetter("question"),
                 "history" : itemgetter("history") | RunnableLambda(history_summarize)} | parallel | context_rekam_jejak
        return chain
    elif result["result"] == "ketentuan_terkait":
        chain = {"question": itemgetter("question"),
                 "history" : itemgetter("history") | RunnableLambda(history_summarize)} | ketentuan_terkait_chain_rag_fusion
        return chain

def rag_fusion_chain():
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

router = router_chain()

history_sum = SummarizerMixin(llm=llm, prompt=SUMMARY_PROMPT)
history_text = "-"

template = RAG_FUSION_PROMPT
prompt_rag_fusion = ChatPromptTemplate.from_template(template)
generate_queries = (
    prompt_rag_fusion 
    | llm
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

rekam_jejak_retriever_mmr = rekam_jejak_vector().as_retriever(search_type="mmr")
rekam_jejak_retriever_sim = rekam_jejak_vector().as_retriever(search_type="similarity")
rekam_jejak_retriever_bm25 = bm25_rekam_retriever()
rekam_jejak_retriever_bm25.k = 10
rekam_jejak_self_retriever = self_retriever_rekam_jejak()

ketentuan_terkait_retriever_mmr = ketentuan_terkait_vector().as_retriever(search_type="mmr")
ketentuan_terkait_retriever_sim = ketentuan_terkait_vector().as_retriever(search_type="similarity")
ketentuan_terkait_retriever_bm25 = bm25_ketentuan_retriever()
ketentuan_terkait_retriever_bm25.k = 10
ketentuan_terkait_self_retriever = self_retriever_ketentuan()

rekam_jejak_graph_chain = graph_rag_chain()
rekam_jejak_chain_rag_fusion = generate_queries | RunnableLambda(multi_retrievers_rekam) | reciprocal_rank_fusion
ketentuan_terkait_chain_rag_fusion = generate_queries | RunnableLambda(multi_retrievers_ketentuan) | reciprocal_rank_fusion

template_rekam_jejak = RAG_REKAM_JEJAK_PROMPT
context_rekam_jejak = PromptTemplate(
    input_variables=["unstructured", "structured"], template=template_rekam_jejak
)

