import streamlit as st
import time
import random
import string

from rag import caller

# sess_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
sess_id = "W56PNA34XM"

# Streamlit page configuration
st.set_page_config(page_title="Chat Application", layout="wide")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to the Sikepo OJK Document Assistant! I'm here to help you navigate and find information on documents available on the Sikepo OJK website. How can I assist you today?"},
         {"role": "assistant", "content": 
'''SIKEPO support 2 types of question: Type 1 is Ketentuan Terkait and Type 2 is Rekam Jejak. To improve the generated answers, you may follow these prompt examples for each question type.\n
Example for Ketentuan Terkait:\n
1.) Jika saya mau membuat produk promosi terkait produk XXX, peraturan apa saja yang perlu saya perhatikan?\n
2.) Apa saja peraturan yang membahas tentang XXX?\n
3.) Sebutkan peraturan-peraturan yang melarang adanya XXX?\n
\n
Example for Rekam Jejak:\n
1.) Apakah peraturan dengan nomor ketentuan XXX masih berlaku?\n
2.) Apakah ada peraturan yang mencabut peraturan XXX?\n
'''}
    ]

if "current_response" not in st.session_state:
    st.session_state.current_response = ""

# We loop through each message in the session state and render it as
# a chat message.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# We take questions/instructions from the chat input to pass to the LLM
if user_prompt := st.chat_input("Your message here", key="user_input"):

    # Add our input to the session state
    st.session_state.messages.append(
        {"role": "user", "content": user_prompt}
    )

    # Add our input to the chat window
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.spinner("Thinking ..."):
        response = caller(user_prompt, sess_id)
    # Add the response to the session state
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    # Add the response to the chat window
    with st.chat_message("assistant"):
        st.markdown(response)