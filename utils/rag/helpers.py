import pickle

from langchain.memory import ConversationSummaryBufferMemory
from utils.constants import CERTAIN_SYSTEM_PROMPT, DOCSTORE_PATH
from utils.llm import get_llm


def format_source_link(file_name, file_link, page_number=None):
    if page_number:
        page_number = int(page_number)
        return f"[{file_name} - Page: {page_number}]({file_link}#page={page_number})"
    return f"[{file_name}]({file_link})"

def format_sources(source_documents):
    sources = {
        format_source_link(
            doc.metadata.get('file_name', ''),
            doc.metadata.get('file_link', ''),
            doc.metadata.get('page_number')
        )
        for doc in source_documents
        if 'file_name' in doc.metadata and 'file_link' in doc.metadata
    }
    return "\n\n".join(sources)


user_memories = {}

def get_user_memory(user_id):
    if user_id not in user_memories:
        llm = get_llm()
        user_memories[user_id] = ConversationSummaryBufferMemory(
            llm=llm,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    return user_memories[user_id]


def is_answer_uncertain(answer):
    llm = get_llm()
    prompt = CERTAIN_SYSTEM_PROMPT.format(answer=answer)
    response = llm.invoke(prompt)
    return "uncertain" in response.content.lower()

def get_docstore():
    docstore_file = open(DOCSTORE_PATH,'rb')
    docstore = pickle.load(docstore_file)
    return docstore
