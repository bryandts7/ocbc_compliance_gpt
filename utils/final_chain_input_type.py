from typing import List, Tuple
from langserve.pydantic_v1 import BaseModel, Field


class InputTypeFinalChain(BaseModel):
    """Input Type for the final chain."""

    chat_history: List[Tuple[str, str]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "question"}},
    )
    question: str