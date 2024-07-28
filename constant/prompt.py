CONTEXTUALIZE_Q_PROMPT_STR = """Given the following conversation and a follow-up question, \
rephrase the follow-up question to be a standalone question in its original language. 
If the follow-up question is not clear, indicate so. 
If the chat history is not relevant to the follow-up question, please ignore the chat history.

Chat History:
{chat_history}

Follow-up Question: {question}
Standalone Question: """