# import nest_asyncio
import streamlit as st
from utils.vector_store import PineconeIndexManager
from utils.model_config import ModelName, get_model
from utils.rag_chain_with_chat_history import create_chain_with_chat_history
from dotenv import load_dotenv
from utils.chat_store import ChatHistory
import hmac
from utils.config import get_config_streamlit
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever


# Apply asyncio and load environment variables
# nest_asyncio.apply()
load_dotenv()

# ============================== FUNCTIONS ==============================


def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.text_input("Password", type="password", help="Input the password to continue",
                  on_change=password_entered, key="password")
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


@st.cache_resource(show_spinner=False)
def load_chain(config: dict = None, top_k: int = 10, top_n: int = 6, model_name: ModelName = ModelName.AZURE_OPENAI, template: str = None, answer_template: str = None):

    llm_model, embed_model = get_model(model_name=model_name, config=config)

    pinecone = PineconeIndexManager(
        index_name='ojk', embed_model=embed_model, config=config)
    vector_store = pinecone.load_vector_index()
    # retriever = vector_store.as_retriever(
    #     search_type="similarity", search_kwargs={"k": top_k})
    compressor = CohereRerank(
        cohere_api_key=config['cohere_api_key'], top_n=top_n)
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": top_k}),
    )
    chain = create_chain_with_chat_history(
        contextualize_q_prompt_str=template,
        qa_system_prompt_str=answer_template,
        retriever=retriever,
        llm_model=llm_model,
    )
    return chain


# ============================== PROMPTING ==============================
_TEMPLATE = """Given the following conversation and a follow-up question, \
rephrase the follow-up question to be a standalone question in its original language. 
If the follow-up question is not clear, indicate so. 
If the chat history is not relevant to the follow-up question, please ignore the chat history.

Chat History:
{chat_history}

Follow-up Question: {question}
Standalone Question: """

_ANSWER_TEMPLATE = """The context information is below.
Context: 
{context}

Based on the context and the metadata information provided, \
answer the query related to banking compliance in Indonesia.
Use the context and metadata information only, without relying on prior knowledge. 
ALWAYS ANSWER IN THE USER'S LANGUAGE.

Please provide your answer in the following format, \
including the regulation number and file URL if available:

(Answer...) \n\n
Source: [metadata['regulation_number']](metadata['file_url'])

If you cannot find the regulation number, just provide the (Answer...). 
If the file_url ends with '.pdf', you can add the metadata['page_number'] in the URL like this: 

(Answer...) \n\n
Source: [metadata['regulation_number']](metadata['file_url#page=metadata['page_number']')

(Answer...) is the answer to the question, don't write '(Answer...)' in the answer.
DO NOT PROVIDE AMBIGUOUS ANSWERS.

Question: {question}
"""


# ============================== CONSTANTS ==============================
TOP_K = 10
TOP_N = 6
model_name = ModelName.AZURE_OPENAI

# ============================== MAIN ==============================

config = get_config_streamlit()

# Page configuration
st.set_page_config(page_title="OJK Chatbot", page_icon="🤖", layout="centered")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
            "content": "Ask me a question about any Regulation of BI and OJK"}
    ]


# Initialize chain and chat history
if "chain" not in st.session_state:
    st.session_state.chain = load_chain(config=config, top_k=TOP_K, top_n=TOP_N, model_name=model_name,
                                        template=_TEMPLATE, answer_template=_ANSWER_TEMPLATE)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatHistory()

if not check_password():
    st.stop()

st.title("Chat with the OJK BOT 💬🤖")

# User input handling
if prompt := st.chat_input("Ask me a question about any Banking Compliance in Indonesia"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        placeholder = st.empty()

        with st.spinner("Generating response..."):
            answer_chunks = []
            for chunk in st.session_state.chain.stream(
                {"chat_history": st.session_state.chat_history, "question": prompt}
            ):
                if "question" in chunk:
                    question = chunk["question"]
                    print(question)
                if "answer" in chunk:
                    answer_chunks.append(chunk["answer"])
                    # Update the placeholder with the current answer
                    placeholder.write("".join(answer_chunks))

            answer = "".join(answer_chunks)
            st.session_state.chat_history.add_chat(prompt, answer)
            message = {"role": "assistant", "content": answer}
            st.session_state.messages.append(message)

# Reset button
if st.button("Reset"):
    st.session_state.chat_history.clear_history()
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about any Regulation of BI and OJK"
        }
    ]
    st.rerun()
