import os

from langchain.chains.conversational_retrieval.base import \
    ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain_pinecone import PineconeVectorStore
from utils.llm import get_embedding_model, get_llm
from utils.utils import is_language_indonesian, translate


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

# Create a dictionary to hold memory objects for each user
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

def get_qa_chain(user_id):
    llm = get_llm()
    embed_model = get_embedding_model()

    # Initialize OpenAI and Pinecone
    PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX", "langchain-test")
    vectorstore = PineconeVectorStore.from_existing_index(PINECONE_INDEX_NAME, embed_model)
    # retriever = vectorstore.as_retriever()

    # Cohere Reranker
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    compressor = CohereRerank()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    # Custom prompt template
    template = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.
    Do not include source citations in your answer. The sources will be added separately.

    Context: {context}
    History: {chat_history}
    Human: {question}
    Assistant: """

    PROMPT = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=template
    )

    # Get or create the user's memory
    memory = get_user_memory(user_id)

    # Create the conversational chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=compression_retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    return qa_chain

def is_answer_uncertain(answer):
    llm = get_llm()

    template = """
    You are an assistant. Determine if the following answer is uncertain or not. 
    Respond with "uncertain" if the answer is uncertain, otherwise respond with "certain".

    Answer: {answer}
    """
    
    prompt = template.format(answer=answer)
    response = llm.invoke(prompt)
    return "uncertain" in response.content.lower()

def caller(question, user_id):
    qa_chain = get_qa_chain(user_id)
    is_indonesian = is_language_indonesian(question)

    if is_indonesian:
        question = translate("en", question)

    result = qa_chain({"question": question})
    answer = result['answer']

    if is_indonesian:
        answer = translate("id", answer)

    if is_answer_uncertain(answer):
        return answer

    formatted_sources = format_sources(result['source_documents'])
    return f"{answer}\n\n\nSources:\n\n{formatted_sources}"