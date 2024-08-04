from langchain.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate
from langchain_community.graphs import Neo4jGraph
from langchain_core.language_models.base import BaseLanguageModel

from constant.sikepo.prompt import GRAPH_QA_GEN_PROMPT, GRAPH_CYPHER_GEN_PROMPT

import dotenv
dotenv.load_dotenv()


def graph_rag_chain(cypher_llm: BaseLanguageModel, qa_llm: BaseLanguageModel, graph: Neo4jGraph):
    qa_generation_template = GRAPH_QA_GEN_PROMPT
    qa_generation_prompt = PromptTemplate(
        input_variables=["context", "question"], template=qa_generation_template)

    cypher_generation_template = GRAPH_CYPHER_GEN_PROMPT
    cypher_generation_prompt = PromptTemplate(input_variables=[
                                              "schema", "question", "history"], template=cypher_generation_template)

    chain = GraphCypherQAChain.from_llm(
        cypher_llm=cypher_llm, qa_llm=qa_llm, graph=graph, verbose=True, qa_prompt=qa_generation_prompt, cypher_prompt=cypher_generation_prompt, validate_cypher=True,
    )
    return chain
