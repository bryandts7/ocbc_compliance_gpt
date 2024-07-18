import os
import dotenv
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings

dotenv.load_dotenv()

# Configuration
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("OPENAI_API_VERSION")

def azure_llm():
    llm = AzureChatOpenAI(
        azure_deployment="gpt-35-crayon",
        api_version=api_version,
        api_key=api_key,
        azure_endpoint=endpoint,
        temperature=0
        # other params...
    )
    return llm

def azure_embeddings():
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="embedding-ada-crayon",
        openai_api_version=api_version,
        api_key=api_key,
        azure_endpoint=endpoint,
    )
    return embeddings