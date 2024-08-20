from typing import List, Optional
import logging
import json
from time import time
import urllib.parse as urlparse

from elasticsearch import Elasticsearch, exceptions as es_exceptions
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict

from langchain_elasticsearch.chat_history import ElasticsearchChatMessageHistory

from langchain_elasticsearch.client import create_elasticsearch_client
from langchain_elasticsearch._utilities import with_user_agent_header

logger = logging.getLogger(__name__)


# ========== ELASTICSEARCH ==========

class CustomElasticsearchChatMessageHistory(ElasticsearchChatMessageHistory):
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
        esnsure_ascii: Optional[bool] = True,
    ):
        self.index: str = index
        self.session_id: str = session_id
        self.ensure_ascii = esnsure_ascii
        self.k = k

        # Initialize Elasticsearch client from passed client arg or connection info
        if es_connection is not None:
            self.client = es_connection
        elif es_url is not None or es_cloud_id is not None:
            try:
                self.client = create_elasticsearch_client(
                    url=es_url,
                    username=es_user,
                    password=es_password,
                    cloud_id=es_cloud_id,
                    api_key=es_api_key,
                )
            except Exception as err:
                logger.error(f"Error connecting to Elasticsearch: {err}")
                raise err
        else:
            raise ValueError(
                """Either provide a pre-existing Elasticsearch connection, \
                or valid credentials for creating a new connection."""
            )

        self.client = with_user_agent_header(self.client, "langchain-py-ms")

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

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the chat session in Elasticsearch"""
        try:
            from elasticsearch import ApiError

            # check if the index exists
            if not self.client.indices.exists(index=self.index):
                logger.debug(
                    f"Creating index {self.index} for storing chat history.")
                self.client.indices.create(
                    index=self.index,
                    mappings={
                        "properties": {
                            "session_id": {"type": "keyword"},
                            "title": {"type": "text"},
                            "created_at": {"type": "date"},
                            "history": {"type": "text"},
                        }
                    },
                )

            # check if the session_id exists, if exists use the title
            query = {
                "query": {
                    "term": {
                        "session_id": self.session_id
                    }
                }
            }
            results = self.client.search(index=self.index, body=query, size=1)
            if results['hits']['total']['value'] == 0:
                title = message_to_dict(message)['data']['content'][:50] if len(message_to_dict(
                    message)['data']['content']) > 50 else message_to_dict(message)['data']['content']
            else:
                title = results['hits']['hits'][0]['_source']['title']

            self.client.index(
                index=self.index,
                document={
                    "session_id": self.session_id,
                    "title": title,
                    "created_at": round(time() * 1000),
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


# ========== CHAT STORE ==========

class ElasticChatStore:
    """A store for managing chat histories using Elasticsearch."""

    def __init__(self, config: dict, k: int = 5):
        self.host = config["es_uri"]
        self.user = config["es_username"]
        self.password = config["es_password"]
        self.index_name = "chat_history2"
        self.k = k

        self.es_client = Elasticsearch(
            hosts=self.host,
            basic_auth=(self.user, self.password),
            max_retries=10,
            verify_certs=False
        )

    def get_session_history(self, user_id: str, conversation_id: str) -> CustomElasticsearchChatMessageHistory:
        session_id = f"{user_id}:{conversation_id}"
        history = CustomElasticsearchChatMessageHistory(
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
        conversation_map = {}
        for hit in results['hits']['hits']:
            session_id = hit['_source']['session_id']
            conversation_id = session_id.split(":")[1]
            if conversation_id not in conversation_map:
                conversation_map[conversation_id] = {
                    "id": urlparse.quote(conversation_id),
                    "title": hit['_source']['title'],
                    "date": hit['_source']['created_at']
                }
            else:
                # Update the date if the current one is more recent
                if hit['_source']['created_at'] > conversation_map[conversation_id]['date']:
                    conversation_map[conversation_id]['date'] = hit['_source']['created_at']
        # Convert the dictionary values to a list
        formatted_result = list(conversation_map.values())
        return formatted_result

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
                "content": message_dict["data"]["content"],
                "date": hit['_source']['created_at']
            }
            messages.append(parsed_message)
        return messages

    def clear_session_history(self, user_id: str, conversation_id: str) -> None:
        """Clear the session history for a given user and conversation."""
        session_id = f"{user_id}:{conversation_id}"
        query = {"query": {"prefix": {"session_id": session_id}}}
        try:
            response = self.es_client.delete_by_query(
                index=self.index_name, body=query)
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

        results = self.es_client.search(
            index=self.index_name, body=query, size=10000)

        if results['hits']['total']['value'] == 0:
            return False

        for hit in results['hits']['hits']:
            doc_id = hit['_id']
            update_body = {
                "doc": {
                    "title": new_title
                }
            }
            self.es_client.update(index=self.index_name,
                                  id=doc_id, body=update_body)

        return True

    def clear_conversation_by_userid(self, user_id: str) -> bool:
        """Clear all conversations for a given user."""
        query = {"query": {"prefix": {"session_id": f"{user_id}:"}}}
        try:
            response = self.es_client.delete_by_query(
                index=self.index_name, body=query, refresh=True)
            return response.get('deleted', 0) > 0
        except Exception as e:
            logger.error(f"Error deleting all chats for user {user_id}: {e}")
            return False
