import random
import string

import streamlit as st
from utils.constants import STREAMLIT_INTRO
from utils.rag.basic_agent import caller

# Set the password
PASSWORD = "bi"

# Streamlit page configuration
st.set_page_config(page_title="Chat Application", layout="wide")

# Initialize login state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Display the login form if not logged in
if not st.session_state.logged_in:
    st.title("Login")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if password == PASSWORD:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Incorrect password. Please try again.")
else:
    # Generate or use a fixed session ID for testing
    if "sess_id" not in st.session_state:
        st.session_state.sess_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
    sess_id = st.session_state.sess_id

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": STREAMLIT_INTRO},
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
