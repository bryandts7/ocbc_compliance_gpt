# model_config.py
from enum import Enum
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI, OpenAIEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings import OllamaEmbeddings

# ================== ENUMS ==================


class ModelName(Enum):
    OLLAMA = 'ollama'
    OPENAI = 'openai'
    AZURE_OPENAI = 'azure_openai'

class LLMModelName(Enum):
    GPT_35_TURBO = 'gpt-3.5-turbo'
    GPT_4O_MINI = 'gpt-4o-mini'

class EmbeddingModelName(Enum):
    EMBEDDING_ADA = 'ada'
    EMBEDDING_3_SMALL = '3-small'

# ================== FUNCTIONS ==================


def get_openai_models(api_key: str, llm_model_name: LLMModelName, embedding_model_name: EmbeddingModelName):
    if llm_model_name == LLMModelName.GPT_35_TURBO:
        llm = ChatOpenAI(
            api_key=api_key,
            temperature=0.0,
            verbose=True,
            model="gpt-3.5-turbo",
        )
    elif llm_model_name == LLMModelName.GPT_4O_MINI:
        llm = ChatOpenAI(
            api_key=api_key,
            temperature=0.0,
            verbose=True,
            model="gpt-4o-mini",
        )
    else:
        # throw error
        raise ValueError("Model not supported in OpenAI")
        
    if embedding_model_name == EmbeddingModelName.EMBEDDING_ADA:
        embedding_llm = OpenAIEmbeddings(
            api_key=api_key,
        )
    elif embedding_model_name == EmbeddingModelName.EMBEDDING_3_SMALL:
        embedding_llm = OpenAIEmbeddings(
            api_key=api_key,
            model="text-embedding-3-small"
        )
    else:
        # throw error
        raise ValueError("Model not supported in OpenAI")

    return llm, embedding_llm


def get_azure_openai_models(azure_endpoint: str, azure_deployment: str, api_version: str, api_key: str, llm_model_name: LLMModelName, embedding_model_name: EmbeddingModelName):
    if llm_model_name == LLMModelName.GPT_35_TURBO:
        llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
            api_key=api_key,
            temperature=0.0,
            verbose=True,
        )
    else:
        # throw error
        raise ValueError("Model not supported in Azure OpenAI")
    
    if embedding_model_name == EmbeddingModelName.EMBEDDING_ADA:
        embedding_llm = AzureOpenAIEmbeddings(
            azure_endpoint=azure_endpoint,
            azure_deployment='embedding-ada-crayon',
            api_key=api_key,
            api_version=api_version,
        )
    elif embedding_model_name == EmbeddingModelName.EMBEDDING_3_SMALL:
        embedding_llm = AzureOpenAIEmbeddings(
            azure_endpoint=azure_endpoint,
            azure_deployment='embedding-3-small',
            api_key=api_key,
            api_version=api_version,
        )
    else:
        # throw error
        raise ValueError("Model not supported in Azure OpenAI")


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


def get_model(model_name: ModelName, config: dict = {}, llm_model_name: LLMModelName = None, embedding_model_name: EmbeddingModelName = None):
    if model_name == ModelName.OPENAI:
        return get_openai_models(**config['config_openai'], llm_model_name=llm_model_name, embedding_model_name=embedding_model_name)
    elif model_name == ModelName.AZURE_OPENAI:
        return get_azure_openai_models(**config['config_azure'], llm_model_name=llm_model_name, embedding_model_name=embedding_model_name)
    elif model_name == ModelName.OLLAMA:
        return get_ollama_models()