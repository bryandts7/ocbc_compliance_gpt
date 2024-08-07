from typing import List, Dict, Tuple, Optional
from langchain.schema import BaseMessage
import logging
import json
import time

from elasticsearch import Elasticsearch, exceptions as es_exceptions
from langchain_community.chat_message_histories import ElasticsearchChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
    messages_from_dict,
)

logger = logging.getLogger(__name__)

# ========== ELASTICSEARCH ==========

class LimitedElasticsearchChatMessageHistory(ElasticsearchChatMessageHistory):
    def __init__(
        self,
        index: str,
        session_id: str,
        k: int = 5,
        *,
        es_connection: Optional["Elasticsearch"] = None,
        es_url: Optional[str] = None,
        es_cloud_id: Optional[str] = None,
        es_user: Optional[str] = None,
        es_api_key: Optional[str] = None,
        es_password: Optional[str] = None,
        ensure_ascii: Optional[bool] = True,
    ):
        self.index: str = index
        self.session_id: str = session_id
        self.ensure_ascii = ensure_ascii
        self.k = k

        # Initialize Elasticsearch client from passed client arg or connection info
        if es_connection is not None:
            self.client = es_connection.options(
                headers={"user-agent": self.get_user_agent()}
            )
        elif es_url is not None or es_cloud_id is not None:
            self.client = ElasticsearchChatMessageHistory.connect_to_elasticsearch(
                es_url=es_url,
                username=es_user,
                password=es_password,
                cloud_id=es_cloud_id,
                api_key=es_api_key,
            )
        else:
            raise ValueError(
                """Either provide a pre-existing Elasticsearch connection, \
                or valid credentials for creating a new connection."""
            )

        if self.client.indices.exists(index=index):
            logger.debug(
                f"Chat history index {index} already exists, skipping creation."
            )
        else:
            logger.debug(f"Creating index {index} for storing chat history.")

            self.client.indices.create(
                index=index,
                mappings={
                    "properties": {
                        "session_id": {"type": "keyword"},
                        "title": {"type": "text"},
                        "created_at": {"type": "date"},
                        "history": {"type": "text"},
                    }
                },
            )

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the chat session in Elasticsearch"""
        try:
            from elasticsearch import ApiError

            # check if the session_id already exists
            result = self.client.search(
                index=self.index,
                query={"term": {"session_id": self.session_id}},
                size=1
            )

            # if the session_id already exists, use the existing title
            if result and len(result["hits"]["hits"]) > 0:
                title = result["hits"]["hits"][0]["_source"]["title"]
            else:
                title = message_to_dict(message)['content'][:25] + '...' if len(message_to_dict(message)['content']) > 25 else message_to_dict(message)['content'] + '...'

            self.client.index(
                index=self.index,
                document={
                    "session_id": self.session_id,
                    "created_at": round(time() * 1000),
                    "title": title,
                    "history": json.dumps(
                        message_to_dict(message),
                        ensure_ascii=bool(self.ensure_ascii),
                    ),
                },
                refresh=True,
            )
        except ApiError as err:
            logger.error(f"Could not add message to Elasticsearch: {err}")
            raise err

    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve the messages from Elasticsearch"""
        try:
            from elasticsearch import ApiError

            result = self.client.search(
                index=self.index,
                query={"term": {"session_id": self.session_id}},
                sort="created_at:desc",
                size=2*self.k
            )
        except ApiError as err:
            logger.error(
                f"Could not retrieve messages from Elasticsearch: {err}")
            raise err

        if result and len(result["hits"]["hits"]) > 0:
            items = [
                json.loads(document["_source"]["history"])
                for document in result["hits"]["hits"]
            ]
        else:
            items = []

        # Retrieve only the last k pairs of messages in the original order
        all_messages = messages_from_dict(items)
        
        # Take the last k Human messages
        last_k_human_ai_messages = all_messages[:self.k]
        last_k_human_ai_messages.reverse()
        return last_k_human_ai_messages


class ElasticChatStore:
    """A store for managing chat histories using Elasticsearch."""

    def __init__(self, config: dict, k: int = 5):
        self.host = config["es_uri"]
        self.user = config["es_username"]
        self.password = config["es_password"]
        self.index_name = "chat_history"
        self.k = k

        self.es_client = Elasticsearch(
            hosts=self.host,
            basic_auth=(self.user, self.password),
            max_retries=10,
            verify_certs=False
        )

    def get_session_history(self, user_id: str, conversation_id: str) -> LimitedElasticsearchChatMessageHistory:
        session_id = f"{user_id}:{conversation_id}"
        history = LimitedElasticsearchChatMessageHistory(
            k=self.k,
            es_connection=self.es_client,
            index=self.index_name,
            session_id=session_id
        )
        return history

    def get_conversation_ids_by_user_id(self, user_id: str) -> List[dict]:
        query = {
            "query": {
                "prefix": {
                    "session_id": f"{user_id}:"
                }
            },
            "sort": [
                {"created_at": {"order": "desc"}}
            ]
        }
        results = self.es_client.search(
            index=self.index_name, body=query, size=10000
        )
        formatted_result = []

        conversation_ids = {}
        for hit in results['hits']['hits']:
            session_id = hit['_source']['session_id']
            _, conversation_id = session_id.split(':', 1)
            
            if conversation_id not in conversation_ids:
                conversation_ids[conversation_id] = hit['_source']['title']

        for conv_id, title in conversation_ids.items():
            formatted_result.append({
                "id": conv_id,
                "title": title
            })

    def get_conversation(self, user_id: str, conversation_id: str) -> List[dict]:
        session_id = f"{user_id}:{conversation_id}"
        query = {
            "query": {
                "term": {
                    "session_id": session_id
                }
            },
            "sort": [
                {"created_at": {"order": "asc"}}
            ]
        }
        results = self.es_client.search(
            index=self.index_name, body=query, size=10000)
        messages = []
        for hit in results['hits']['hits']:
            message_str = hit['_source']['history']
            message_dict = json.loads(message_str)
            parsed_message = {
                "isUser": message_dict["type"] == "human",
                "content": message_dict["data"]["content"]
            }
            messages.append(parsed_message)
        return messages

    def clear_session_history(self, user_id: str, conversation_id: str) -> None:
        """Clear the session history for a given user and conversation."""
        session_id = f"{user_id}:{conversation_id}"
        query = {"query": {"prefix": {"session_id": session_id}}}
        try:
            response = self.es_client.delete_by_query(index=self.index_name, body=query)
            return response.get('deleted', 0) > 0
        except Exception as e:
            return False


    def clear_session_history_by_userid(self, user_id: str) -> None:
        """Clear the session history for a given user."""
        query = {"query": {"prefix": {"session_id": f"{user_id}:"}}}
        self.es_client.delete_by_query(index=self.index_name, body=query)

    def clear_all(self) -> None:
        """Clear all session histories."""
        try:
            self.es_client.indices.delete(index=self.index_name)
        except es_exceptions.NotFoundError:
            pass  # Index doesn't exist, so no need to delete

    def rename_title(self, user_id: str, conversation_id: str, new_title: str) -> bool:
        session_id = f"{user_id}:{conversation_id}"
    
        query = {
            "query": {
                "term": {
                    "session_id": session_id
                }
            }
        }
        
        results = self.es_client.search(index=self.index_name, body=query, size=10000)

        if results['hits']['total']['value'] == 0:
            return False

        for hit in results['hits']['hits']:
            doc_id = hit['_id']
            update_body = {
                "doc": {
                    "title": new_title
                }
            }
            self.es_client.update(index=self.index_name, id=doc_id, body=update_body)
        
        return True