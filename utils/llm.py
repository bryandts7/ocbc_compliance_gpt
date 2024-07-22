import os

from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_community.chat_models.azure_openai import AzureChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_community.embeddings.azure_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings

# Load environment variables from .env file
load_dotenv()

def get_llm(company="azure_openai", model="gpt-35-crayon"):
    llm = None
    if company == "azure_openai":
        api_key = os.getenv('AZURE_OPENAI_API_KEY')
        api_version = os.getenv("OPENAI_API_VERSION")
        api_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

        llm = AzureChatOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=api_endpoint,
            azure_deployment=model,
            temperature=0,
        )

    elif company == "openai":
        api_key = os.getenv('OPENAI_API_KEY')

        if not model:
            model = "gpt-4o-mini"

        llm = ChatOpenAI(api_key=api_key, model=model, temperature=0)
    
    return llm

def get_embedding_model(company="azure_openai", model="embedding-ada-crayon"):
    embed_model = None
    if company == "azure_openai":
        api_key = os.getenv('AZURE_OPENAI_API_KEY')
        api_version = os.getenv("OPENAI_API_VERSION")
        api_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

        embed_model = AzureOpenAIEmbeddings(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=api_endpoint,
            azure_deployment=model,
        )

    elif company == "openai":
        api_key = os.getenv('OPENAI_API_KEY')

        if not model:
            model = "text-embedding-3-small"

        embed_model = OpenAIEmbeddings(
            api_key=api_key,
            model=model,
        )
    
    return embed_model

# Example usage
if __name__ == "__main__":
    llm = get_llm(company="azure_openai", model="gpt-35-crayon")
    print(f"LLM: {llm}")

    embedding_model = get_embedding_model(company="azure_openai", model="embedding-ada-crayon")
    print(f"Embedding Model: {embedding_model}")
