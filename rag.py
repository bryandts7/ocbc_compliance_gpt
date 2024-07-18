import warnings
from langchain_core._api.beta_decorator import LangChainBetaWarning
warnings.filterwarnings("ignore", category=LangChainBetaWarning)

from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from utils.conversation import CustomChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableLambda
from retry import retry

from utils.azure_openai import azure_llm, azure_embeddings
from constants.prompt import RAG_PROMPT
from handler.rag_fusion import rag_fusion_chain
from handler.lotr import lotr_context_chain

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = CustomChatMessageHistory()
    return store[session_id]

def get_source_docs(ls_tup):
    new_ls = []
    for tup in ls_tup:
        new_ls.append(tup[0])
    return new_ls

store = {}
llm = azure_llm()
embeddings = azure_embeddings()

# RAG
system_prompt = RAG_PROMPT
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

context_chain = lotr_context_chain()
full_chain = context_chain | prompt | llm | StrOutputParser()

full_chain_with_context = context_chain | {"query":itemgetter("question"), "result": prompt | llm | StrOutputParser(), "source_documents": itemgetter("context") } #| RunnableLambda(get_source_docs)}

full_chain_with_message_history = RunnableWithMessageHistory(
    full_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

full_chain_with_context_and_message_history = RunnableWithMessageHistory(
    full_chain_with_context,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

def chain_with_context():
    return full_chain_with_context

# Function to invoke the full chain
@retry(tries=3, delay=2)
def caller(message, sess_id):
    response = full_chain_with_message_history.invoke(
        {"question": message},
        config={"configurable": {"session_id": sess_id}})
    return response


def caller_with_context(message, sess_id):
    # print(sess_id)
    # print("Store:", store)
    response = full_chain_with_context_and_message_history.invoke(
        {"question": message},
        config={"configurable": {"session_id": sess_id}})
    return response