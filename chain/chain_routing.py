from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.language_models.base import BaseLanguageModel

from constant.sikepo.prompt import ROUTER_PROMPT, KETENTUAN_ANSWERING_PROMPT

# ====== CHAIN ROUTING ======

# Query of routing


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["rekam_jejak", "ketentuan_terkait", ] = Field(
        ...,
        description="Given a user question choose which datasource would be most relevant for answering their question",
    )


def get_string_routing(route):
    return route.datasource


# Query of answering
class RouteQueryAnswer(BaseModel):
    """Route an LLM Response whether it is answering the question or not."""

    decision: Literal["YES", "NO", ] = Field(
        ...,
        description="Given an LLM Response to the Question, choose whether the response is answering the question or NOT answering the question",
    )


def get_string_answer(route):
    return route.decision

# Router chain


def question_router_chain(llm_model: BaseLanguageModel, ROUTING_PROMPT: str = ROUTER_PROMPT):
    llm = llm_model
    structured_llm = llm.with_structured_output(RouteQuery)
    system = ROUTING_PROMPT
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            # MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    # Define router
    router = prompt | structured_llm | RunnableLambda(get_string_routing)
    return router


def ketentuan_router_chain(llm_model: BaseLanguageModel, ROUTING_PROMPT: str = KETENTUAN_ANSWERING_PROMPT):
    llm = llm_model
    structured_llm = llm.with_structured_output(RouteQueryAnswer)
    system = ROUTING_PROMPT
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            # MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
            ("ai", "{answer}")
        ]
    )

    # Define router
    router = prompt | structured_llm | RunnableLambda(get_string_answer)
    return router
