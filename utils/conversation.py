from langchain_core.chat_history import BaseChatMessageHistory
from typing import List, Sequence
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage

class CustomChatMessageHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history.

    Stores messages in an in memory list.
    """

    messages: List[BaseMessage] = Field(default_factory=list)
    max_messages: int = 4  # Set the limit (K) of messages to keep

    async def aget_messages(self) -> List[BaseMessage]:
        return self.messages

    def add_message(self, message: BaseMessage) -> None:
        """Add a self-created message to the store"""
        self.messages.append(message)
        # Keep last k messages
        self.messages = self.messages[-self.max_messages:]


    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add messages to the store"""
        self.add_messages(messages)

    def clear(self) -> None:
        self.messages = []

    async def aclear(self) -> None:
        self.clear()
