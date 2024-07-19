# model_config.py
from enum import Enum
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI, OpenAIEmbeddings
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

# ================== ENUMS ==================


class ModelName(Enum):
    OLLAMA = 'ollama'
    OPENAI = 'openai'
    AZURE_OPENAI = 'azure_openai'

# ================== FUNCTIONS ==================


def get_openai_models(api_key: str):
    llm = ChatOpenAI(
        api_key=api_key,
        temperature=0.0,
        verbose=True,
        model="gpt-3.5-turbo",
    )
    embedding_llm = OpenAIEmbeddings(
        api_key=api_key,
    )
    return llm, embedding_llm


def get_azure_openai_models(azure_endpoint: str, azure_deployment: str, api_version: str, api_key: str):
    llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment,
        api_version=api_version,
        api_key=api_key,
        temperature=0.0,
        verbose=True,
    )
    embedding_llm = AzureOpenAIEmbeddings(
        azure_endpoint=azure_endpoint,
        azure_deployment='embedding-ada-crayon',
        api_key=api_key,
        api_version=api_version,
    )
    return llm, embedding_llm


def get_ollama_models():
    llm = Ollama(
        model='llama3'
    )
    embedding_llm = OllamaEmbeddings(
        model='llama3'
    )
    return llm, embedding_llm

# ================== MAIN ==================


def get_model(model_name: ModelName, config: dict = {}):
    if model_name == ModelName.OPENAI:
        return get_openai_models(**config['config_openai'])
    elif model_name == ModelName.AZURE_OPENAI:
        return get_azure_openai_models(**config['config_azure'])
    elif model_name == ModelName.OLLAMA:
        return get_ollama_models()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
