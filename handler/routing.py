from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from utils.azure_openai import azure_llm
from constants.prompt import ROUTER_PROMPT

# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["rekam_jejak", "ketentuan_terkait", ] = Field(
        ...,
        description="Given a user question choose which datasource would be most relevant for answering their question",
    )

def get_string(route):
    return route.datasource

def router_chain():
    llm = azure_llm()
    structured_llm = llm.with_structured_output(RouteQuery)
    system = ROUTER_PROMPT
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            # MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    # Define router 
    router = prompt | structured_llm | RunnableLambda(get_string)

    return router