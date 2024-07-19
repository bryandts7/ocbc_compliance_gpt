from pymongo import MongoClient
from typing import List, Dict, Tuple
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_mongodb import MongoDBChatMessageHistory


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



class ChatStore:
    """A store for managing chat histories using MongoDB."""

    def __init__(self, config: dict, k: int = 5):
        self.client = MongoClient(config["mongo_uri"])
        self.db = self.client["chat_history"]
        self.uri = config["mongo_uri"]
        self.k = k

    def get_session_history(self, user_id: str, conversation_id: str) -> MongoDBChatMessageHistory:
        collection_name = f"{user_id}_{conversation_id}"
        session_id = f"{user_id}:{conversation_id}"
        # history = MongoDBChatMessageHistory(
        #     connection_string=self.uri,
        #     database_name=self.db.name,
        #     collection_name=collection_name,
        #     session_id=session_id,
        # )
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


# class InMemoryHistory(BaseChatMessageHistory, BaseModel):
#     """In memory implementation of chat message history."""

#     messages: List[BaseMessage] = Field(default_factory=list)
#     k: int = 5 # last k messages

#     def add_message(self, message: BaseMessage) -> None:
#         """Add a self-created message to the store"""
#         self.messages.append(message)
#         self.messages = self.messages[-self.k:]

#     def clear(self) -> None:
#         self.messages = []


# class ChatStore:
#     """A store for managing chat histories."""

#     def __init__(self, k: int = 3):
#         self.store: Dict[Tuple[str, str], InMemoryHistory] = {}
#         self.k = k

#     def get_session_history(self, user_id: str, conversation_id: str) -> BaseChatMessageHistory:
#         if (user_id, conversation_id) not in self.store:
#             self.store[(user_id, conversation_id)] = InMemoryHistory(k=self.k)
#         return self.store[(user_id, conversation_id)]

#     def clear_session_history(self, user_id: str, conversation_id: str) -> None:
#         """Clear the session history for a given user and conversation."""
#         if (user_id, conversation_id) in self.store:
#             self.store[(user_id, conversation_id)].clear()

#     def add_message_to_history(self, user_id: str, conversation_id: str, message: BaseMessage) -> None:
#         """Add a message to the session history for a given user and conversation."""
#         history = self.get_session_history(user_id, conversation_id)
#         history.add_message(message)

#     def clear_all(self) -> None:
#         """Clear all session histories."""
#         for history in self.store.values():
#             history.clear()

#     def get_all_history(self) -> Dict[Tuple[str, str], InMemoryHistory]:
#         return self.store


# from typing import List, Tuple


# class ChatHistory:
#     """Stores and formats chat history."""

#     def __init__(self, max_history_length: int = 10):
#         self.max_history_length = max_history_length
#         self.history = []

#     def add_chat(self, human_message: str, assistant_message: str) -> None:
#         """Add a dialogue to the chat history."""
#         self.history.append((human_message, assistant_message))
#         if len(self.history) > self.max_history_length:
#             self.history = self.history[-self.max_history_length:]

#     def get_history(self) -> List[Tuple[str, str]]:
#         """Retrieve the entire chat history."""
#         return self.history

#     def get_formatted_history(self) -> str:
#         """Format chat history into a readable string."""
#         formatted_history = ""
#         for dialogue_turn in self.history:
#             human = "Human: " + dialogue_turn[0]
#             assistant = "Assistant: " + dialogue_turn[1]
#             formatted_history += 10*"-" + "\n" + human + "\n" + 10*"-" + "\n" + assistant + "\n"
#         return formatted_history

#     def clear_history(self) -> None:
#         """Clear the chat history."""
#         self.history = []
#         print("Chat history cleared.")
