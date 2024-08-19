
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
# for local testing


def get_config():
    openai_api_key = os.getenv("OPENAI_KEY")

    azure_api_key_llm = os.getenv("AZURE_OPENAI_KEY_LLM")
    azure_api_version_llm = os.getenv("AZURE_API_VERSION_LLM")
    azure_api_endpoint_llm = os.getenv("AZURE_OPENAI_ENDPOINT_LLM")
    azure_api_deployment_id_llm = os.getenv("AZURE_OPENAI_DEPLOYMENT_ID_LLM")

    azure_api_key_emb = os.getenv("AZURE_OPENAI_KEY_EMB")
    azure_api_version_emb = os.getenv("AZURE_API_VERSION_EMB")
    azure_api_endpoint_emb = os.getenv("AZURE_OPENAI_ENDPOINT_EMB")
    azure_api_deployment_id_emb = os.getenv("AZURE_OPENAI_DEPLOYMENT_ID_EMB")

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    cohere_api_key = os.getenv("COHERE_API_KEY")
    mongo_uri = os.getenv("MONGO_URI")
    redis_uri = os.getenv("REDIS_URI")
    neo4j_uri = os.getenv("NEO4J_GRAPH_URL")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    neo4j_db = os.getenv("NEO4J_DATABASE")
    postgres_uri = os.getenv("POSTGRES_URI")
    es_uri = os.getenv("ES_URI")
    es_username = os.getenv("ES_USERNAME")
    es_password = os.getenv("ES_PASSWORD")

    config_openai = {
        'api_key': openai_api_key,
    }

    config_azure_llm = {
        'azure_endpoint': azure_api_endpoint_llm,
        'azure_deployment': azure_api_deployment_id_llm,
        'api_version': azure_api_version_llm,
        'api_key': azure_api_key_llm
    }

    config_azure_emb = {
        'azure_endpoint': azure_api_endpoint_emb,
        'azure_deployment': azure_api_deployment_id_emb,
        'api_version': azure_api_version_emb,
        'api_key': azure_api_key_emb
    }

    return {
        "config_openai": config_openai,
        "config_azure_llm": config_azure_llm,
        "config_azure_emb": config_azure_emb,
        "pinecone_api_key": pinecone_api_key,
        "cohere_api_key": cohere_api_key,
        "postgres_uri": postgres_uri,
        "mongo_uri": mongo_uri,
        "redis_uri": redis_uri,
        "neo4j_uri": neo4j_uri,
        "neo4j_username": neo4j_username,
        "neo4j_password": neo4j_password,
        "neo4j_db": neo4j_db,
        "es_uri": es_uri,
        "es_username": es_username,
        "es_password": es_password
    }

# for deployment


def get_config_streamlit():
    openai_api_key = st.secrets["openai_key"]
    azure_api_key = st.secrets["azure_openai_key"]
    azure_api_version = st.secrets["api_version"]
    azure_api_endpoint = st.secrets["azure_openai_endpoint"]
    azure_api_deployment_id = st.secrets["azure_openai_deployment_id"]
    pinecone_api_key = st.secrets["pinecone_api_key"]
    cohere_api_key = st.secrets["cohere_api_key"]
    mongo_uri = st.secrets["mongo_uri"]
    redis_uri = st.secrets["redis_uri"]
    postgres_uri = st.secrets["postgres_uri"]
    neo4j_uri = st.secrets["neo4j_graph_url"]
    neo4j_username = st.secrets["neo4j_username"]
    neo4j_password = st.secrets["neo4j_password"]
    neo4j_db = st.secrets["neo4j_database"]
    es_uri = st.secrets["es_uri"]
    es_username = st.secrets["es_username"]
    es_password = st.secrets["es_password"]

    config_openai = {
        'api_key': openai_api_key,
    }

    config_azure = {
        'azure_endpoint': azure_api_endpoint,
        'azure_deployment': azure_api_deployment_id,
        'api_version': azure_api_version,
        'api_key': azure_api_key
    }

    return {
        "config_openai": config_openai,
        "config_azure": config_azure,
        "pinecone_api_key": pinecone_api_key,
        "cohere_api_key": cohere_api_key,
        "mongo_uri": mongo_uri,
        "redis_uri": redis_uri,
        "postgres_uri": postgres_uri,
        "neo4j_uri": neo4j_uri,
        "neo4j_username": neo4j_username,
        "neo4j_password": neo4j_password,
        "neo4j_db": neo4j_db,
        "es_uri": es_uri,
        "es_username": es_username,
        "es_password": es_password
    }
