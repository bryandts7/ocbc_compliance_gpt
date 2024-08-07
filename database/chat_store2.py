from typing import List, Dict, Tuple, Optional
from langchain.schema import BaseMessage
import time
import logging
import json

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
        user_id: str,
        conversation_id: str,
        title: str = None,
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
        self.user_id: str = user_id
        self.conversation_id: str = conversation_id
        self.ensure_ascii = ensure_ascii
        self.title = title

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
                        "user_id": {"type": "keyword"},
                        "conversation_id": {"type": "keyword"},
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

            self.client.index(
                index=self.index,
                document={
                    "session_id": self.session_id,
                    "user_id": self.user_id,
                    "conversation_id": self.conversation_id,
                    "title": message_to_dict(message)['content'][:25] + '...' if len(message_to_dict(message)['content']) > 25 else message_to_dict(message)['content'] + '...',
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
        self.index_name = "chat_history_temp"
        self.k = k

        self.es_client = Elasticsearch(
            hosts=self.host,
            basic_auth=(self.user, self.password),
            max_retries=10,
            verify_certs=False
        )

    def get_session_history(self, user_id: str, conversation_id: str) -> LimitedElasticsearchChatMessageHistory:
        history = LimitedElasticsearchChatMessageHistory(
            k=self.k,
            es_connection=self.es_client,
            index=self.index_name,
            user_id=user_id,
            conversation_id=conversation_id,
            session_id=f"{user_id}:{conversation_id}",
        )
        return history

    def get_conversation_ids_by_user_id(self, user_id: str) -> List[dict]:
        query = {
            "query": {
                "term": {
                    "user_id": user_id
                }
            },
            "sort": [
                {"created_at": {"order": "desc"}}
            ]
        }
        results = self.es_client.search(
            index=self.index_name, body=query, size=10000
        )
        conversation_ids = {}
        for hit in results['hits']['hits']:
            conversation_id = hit['_source']['conversation_id']
            if conversation_id not in conversation_ids:
                conversation_ids[conversation_id] = hit['_source']['history']
            else:
                # Update to the most recent message
                conversation_ids[conversation_id] = hit['_source']['history']

        # Format result
        formatted_result = []
        for conv_id, last_message_str in conversation_ids.items():
            last_message = json.loads(last_message_str)
            content = last_message["data"]["content"]
            title = (content[:25] + '...') if len(content) > 25 else content
            formatted_result.append({
                "id": conv_id,
                "title": title
            })

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

    def add_message_to_history(self, user_id: str, conversation_id: str, message: BaseMessage) -> None:
        """Add a message to the session history for a given user and conversation."""
        history = self.get_session_history(user_id, conversation_id)
        history.add_message(message)

    def clear_all(self) -> None:
        """Clear all session histories."""
        try:
            self.es_client.indices.delete(index=self.index_name)
        except es_exceptions.NotFoundError:
            pass  # Index doesn't exist, so no need to delete

    def get_all_history(self) -> Dict[Tuple[str, str], List[BaseMessage]]:
        all_history = {}
        query = {"query": {"match_all": {}},
                 "size": 10000}  # Adjust size as needed
        results = self.es_client.search(index=self.index_name, body=query)

        for hit in results['hits']['hits']:
            session_id = hit['_source']['session_id']
            user_id, conversation_id = session_id.split(':', 1)
            history = self.get_session_history(user_id, conversation_id)
            all_history[(user_id, conversation_id)] = history.messages

        return all_history

    def rename_conversation(self, user_id: str, old_conversation_id: str, new_conversation_id: str) -> bool:
        old_session_id = f"{user_id}:{old_conversation_id}"
        new_session_id = f"{user_id}:{new_conversation_id}"
    
        query = {
            "query": {
                "term": {
                    "session_id": old_session_id
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
                    "session_id": new_session_id
                }
            }
            self.es_client.update(index=self.index_name, id=doc_id, body=update_body)
        
        return True