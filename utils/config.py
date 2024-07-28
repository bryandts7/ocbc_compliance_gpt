
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
# for local testing
def get_config():
    openai_api_key = os.getenv("OPENAI_KEY")
    azure_api_key = os.getenv("AZURE_OPENAI_KEY")
    azure_api_version = os.getenv("AZURE_API_VERSION")
    azure_api_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_deployment_id = os.getenv("AZURE_OPENAI_DEPLOYMENT_ID")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    cohere_api_key = os.getenv("COHERE_API_KEY")
    mongo_uri = os.getenv("MONGO_URI")
    redis_uri = os.getenv("REDIS_URI")
    neo4j_uri = os.getenv("NEO4J_GRAPH_URL")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    neo4j_db = os.getenv("NEO4J_DATABASE")


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
        "neo4j_uri": neo4j_uri,
        "neo4j_username": neo4j_username,
        "neo4j_password": neo4j_password,
        "neo4j_db": neo4j_db
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
    neo4j_uri = st.secrets["neo4j_graph_url"]
    neo4j_username = st.secrets["neo4j_username"]
    neo4j_password = st.secrets["neo4j_password"]
    neo4j_db = st.secrets["neo4j_database"]

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
        "neo4j_uri": neo4j_uri,
        "neo4j_username": neo4j_username,
        "neo4j_password": neo4j_password,
        "neo4j_db": neo4j_db
    }

