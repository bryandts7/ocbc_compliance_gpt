OCR_CONFIDENCE_THRESHOLD = 0.90

DOCSTORE_PATH = "pickles/docstore.pkl"

STREAMLIT_INTRO = "Welcome to the Bank Indonesia Document Assistant! I'm here to help you navigate and find information on documents available on the Bank Indonesia website. How can I assist you today?"

MAIN_SYSTEM_PROMPT = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, say that you don't know. Do not include source citations in your answer. The sources will be added separately.

    Context: {context}
    History: {chat_history}
    Human: {question}
    Assistant: """

CERTAIN_SYSTEM_PROMPT = """
    You are an assistant. Determine if the following answer is uncertain or not. 
    Respond with "uncertain" if the answer is uncertain, otherwise respond with "certain".

    Answer: {answer}
    """
