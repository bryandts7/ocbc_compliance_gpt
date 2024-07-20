from pymongo import MongoClient
from typing import List, Dict, Tuple
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_mongodb import MongoDBChatMessageHistory
import redis
from langchain_community.chat_message_histories import RedisChatMessageHistory


# ========== REDIS ==========
class LimitedRedisChatMessageHistory(RedisChatMessageHistory):
    def __init__(self, k: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k

    def add_message(self, message: BaseMessage) -> None:
        # Directly add the message without limiting
        super().add_message(message)

    @property
    def messages(self) -> List[BaseMessage]:
        # Retrieve only the last k pairs of messages in the original order
        all_messages = super().messages
        message_pairs = []
        for i in range(0, len(all_messages) - 1, 2):
            if isinstance(all_messages[i], HumanMessage) and isinstance(all_messages[i+1], AIMessage):
                message_pairs.append((all_messages[i], all_messages[i+1]))

        # Take the last k pairs
        last_k_pairs = message_pairs[-self.k:]

        # Flatten the pairs back into a list
        limited_messages = [msg for pair in last_k_pairs for msg in pair]

        return limited_messages


class RedisChatStore:
    """A store for managing chat histories using Redis."""

    def __init__(self, config: dict, k: int = 5):
        self.redis_uri = config["redis_uri"] + "/1"
        self.redis_client = redis.from_url(self.redis_uri)
        self.k = k

    def get_session_history(self, user_id: str, conversation_id: str) -> LimitedRedisChatMessageHistory:
        session_id = f"{user_id}:{conversation_id}"
        history = LimitedRedisChatMessageHistory(
            k=self.k,
            url=self.redis_uri,
            session_id=session_id,
            key_prefix="chat_history:",
        )
        return history

    def clear_session_history(self, user_id: str, conversation_id: str) -> None:
        """Clear the session history for a given user and conversation."""
        session_id = f"{user_id}:{conversation_id}"
        self.redis_client.delete(f"chat_history:{session_id}")

    def clear_session_history_by_userid(self, user_id: str) -> None:
        """Clear the session history for a given user."""
        pattern = f"chat_history:{user_id}:*"
        keys = self.redis_client.keys(pattern)
        if keys:
            self.redis_client.delete(*keys)

    def add_message_to_history(self, user_id: str, conversation_id: str, message: BaseMessage) -> None:
        """Add a message to the session history for a given user and conversation."""
        history = self.get_session_history(user_id, conversation_id)
        history.add_message(message)

    def clear_all(self) -> None:
        """Clear all session histories."""
        pattern = "chat_history:*"
        keys = self.redis_client.keys(pattern)
        if keys:
            self.redis_client.delete(*keys)

    def get_all_history(self) -> Dict[Tuple[str, str], List[BaseMessage]]:
        all_history = {}
        pattern = "chat_history:*"
        keys = self.redis_client.keys(pattern)
        for key in keys:
            session_id = key.decode('utf-8').split(':')[1:]
            user_id, conversation_id = session_id
            history = self.get_session_history(user_id, conversation_id)
            all_history[(user_id, conversation_id)] = history.messages
        return all_history


# ========== MONGODB ==========
class LimitedMongoDBChatMessageHistory(MongoDBChatMessageHistory):
    def __init__(self, k: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k

    def add_message(self, message: BaseMessage) -> None:
        # Directly add the message without limiting
        super().add_message(message)

    @property
    def messages(self) -> List[BaseMessage]:
        # Retrieve only the last k pairs of messages in the original order
        all_messages = super().messages
        message_pairs = []
        for i in range(0, len(all_messages) - 1, 2):
            if isinstance(all_messages[i], HumanMessage) and isinstance(all_messages[i+1], AIMessage):
                message_pairs.append((all_messages[i], all_messages[i+1]))

        # Take the last k pairs
        last_k_pairs = message_pairs[-self.k:]

        # Flatten the pairs back into a list
        limited_messages = [msg for pair in last_k_pairs for msg in pair]

        return limited_messages


class MongoDBChatStore:
    """A store for managing chat histories using MongoDB."""

    def __init__(self, config: dict, k: int = 5):
        self.uri = config["mongo_uri"]
        self.client = MongoClient(self.uri)
        self.db = self.client["chat_history"]
        self.k = k

    def get_session_history(self, user_id: str, conversation_id: str) -> MongoDBChatMessageHistory:
        collection_name = f"{user_id}_{conversation_id}"
        session_id = f"{user_id}:{conversation_id}"
        history = LimitedMongoDBChatMessageHistory(
            k=self.k,
            connection_string=self.uri,
            database_name=self.db.name,
            collection_name=collection_name,
            session_id=session_id,
        )
        return history

    def clear_session_history(self, user_id: str, conversation_id: str) -> None:
        """Clear the session history for a given user and conversation."""
        collection_name = f"{user_id}_{conversation_id}"
        self.db[collection_name].drop()

    def clear_session_history_by_userid(self, user_id: str) -> None:
        """Clear the session history for a given user."""
        for collection_name in self.db.list_collection_names():
            if collection_name.startswith(user_id):
                self.db[collection_name].drop()

    def add_message_to_history(self, user_id: str, conversation_id: str, message: BaseMessage) -> None:
        """Add a message to the session history for a given user and conversation."""
        history = self.get_session_history(user_id, conversation_id)
        history.add_message(message)

    def clear_all(self) -> None:
        """Clear all session histories."""
        for collection_name in self.db.list_collection_names():
            # self.db[collection_name].delete_many({})
            self.db[collection_name].drop()

    def get_all_history(self) -> Dict[Tuple[str, str], List[BaseMessage]]:
        all_history = {}
        for collection_name in self.db.list_collection_names():
            user_id, conversation_id = collection_name.split('_', 1)
            history = self.get_session_history(user_id, conversation_id)
            all_history[(user_id, conversation_id)] = history.messages
        return all_history
