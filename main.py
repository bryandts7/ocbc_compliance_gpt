from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from pydantic import BaseModel
from utils.config import get_config
from utils.models import ModelName, LLMModelName, EmbeddingModelName, get_model
from database.vector_store.vector_store import ElasticIndexManager, RedisIndexManager
from database.vector_store.neo4j_graph_store import Neo4jGraphStore
from retriever.retriever_ojk.retriever_ojk import get_retriever_ojk
from retriever.retriever_bi.retriever_bi import get_retriever_bi
from retriever.retriever_sikepo.lotr_sikepo import lotr_sikepo
from database.chat_store import ElasticChatStore, RedisChatStore
from chain.rag_chain import create_combined_answer_chain, create_chain_with_chat_history, print_answer_stream
from chain.chain_sikepo.graph_cypher_sikepo_chain import graph_rag_chain
from chain.rag_chain import get_response
from typing import AsyncGenerator

import nest_asyncio
import warnings

nest_asyncio.apply()
warnings.filterwarnings("ignore")

# =========== CONFIG ===========
security = HTTPBearer()

config = get_config()

CONVERSATION_ID = 'conv_1'

# =========== GLOBAL VARIABLES ===========
llm_model, embed_model = get_model(model_name=ModelName.AZURE_OPENAI, config=config,
                                   llm_model_name=LLMModelName.GPT_35_TURBO, embedding_model_name=EmbeddingModelName.EMBEDDING_3_SMALL)
top_n = 5

# index_ojk = RedisIndexManager(index_name='ojk', embed_model=embed_model, config=config, db_id=0)
# index_bi = RedisIndexManager(index_name='bi', embed_model=embed_model, config=config, db_id=0)
# index_sikepo_ket = RedisIndexManager(index_name='sikepo-ketentuan-terkait', embed_model=embed_model, config=config, db_id=0)
# index_sikepo_rek = RedisIndexManager(index_name='sikepo-rekam-jejak', embed_model=embed_model, config=config, db_id=0)
index_ojk = ElasticIndexManager(
    index_name='ojk', embed_model=embed_model, config=config)
index_bi = ElasticIndexManager(
    index_name='bi', embed_model=embed_model, config=config)
index_sikepo_ket = ElasticIndexManager(
    index_name='sikepo-ketentuan-terkait', embed_model=embed_model, config=config)
index_sikepo_rek = ElasticIndexManager(
    index_name='sikepo-rekam-jejak', embed_model=embed_model, config=config)

vector_store_ojk = index_ojk.load_vector_index()
vector_store_bi = index_bi.load_vector_index()
vector_store_ket = index_sikepo_ket.load_vector_index()
vector_store_rek = index_sikepo_rek.load_vector_index()

neo4j_sikepo = Neo4jGraphStore(config=config)
graph = neo4j_sikepo.get_graph()

# chat_store = RedisChatStore(k=3, config=config)
chat_store = ElasticChatStore(k=3, config=config)

# Initialize retrievers and chains with default model
retriever_ojk = get_retriever_ojk(vector_store=vector_store_ojk, top_n=top_n,
                                  llm_model=llm_model, embed_model=embed_model, config=config)
retriever_bi = get_retriever_bi(vector_store=vector_store_bi, top_n=top_n,
                                llm_model=llm_model, embed_model=embed_model, config=config)
retriever_sikepo_ket = lotr_sikepo(vector_store=vector_store_ket, top_n=top_n,
                                   llm_model=llm_model, embed_model=embed_model, config=config)
retriever_sikepo_rek = lotr_sikepo(vector_store=vector_store_rek, top_n=top_n,
                                   llm_model=llm_model, embed_model=embed_model, config=config)

graph_chain = graph_rag_chain(llm_model, llm_model, graph=graph)
chain = create_combined_answer_chain(
    llm_model=llm_model,
    graph_chain=graph_chain,
    retriever_ojk=retriever_ojk,
    retriever_bi=retriever_bi,
    retriever_sikepo_ketentuan=retriever_sikepo_ket,
    retriever_sikepo_rekam=retriever_sikepo_rek,
)

chain_history = create_chain_with_chat_history(
    final_chain=chain,
    chat_store=chat_store,
)

# =========== FASTAPI APP ===========
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    user_input: str


class ChatResponse(BaseModel):
    user_message: str
    ai_response: str


class ModelRequest(BaseModel):
    model: str


@app.post("/initialize_model/")
async def initialize_model(request: ModelRequest):
    global llm_model, embed_model, retriever_ojk, retriever_bi, retriever_sikepo_ket, retriever_sikepo_rek, chain, chain_history, top_n

    model = request.model
    print(f"Received model: {model}")

    if model == 'GPT_4O_MINI':
        llm_model, embed_model = get_model(model_name=ModelName.OPENAI, config=config,
                                           llm_model_name=LLMModelName.GPT_4O_MINI, embedding_model_name=EmbeddingModelName.EMBEDDING_3_SMALL)
        top_n = 10
    elif model == 'GPT_35_TURBO':
        llm_model, embed_model = get_model(model_name=ModelName.AZURE_OPENAI, config=config,
                                           llm_model_name=LLMModelName.GPT_35_TURBO, embedding_model_name=EmbeddingModelName.EMBEDDING_3_SMALL)
        top_n = 5
    else:
        raise HTTPException(status_code=400, detail="Invalid model specified")

    # Reinitialize retrievers with the new model and top_n
    retriever_ojk = get_retriever_ojk(vector_store=vector_store_ojk, top_n=top_n,
                                      llm_model=llm_model, embed_model=embed_model, config=config)
    retriever_bi = get_retriever_bi(vector_store=vector_store_bi, top_n=top_n,
                                    llm_model=llm_model, embed_model=embed_model, config=config)
    retriever_sikepo_ket = lotr_sikepo(vector_store=vector_store_ket, top_n=top_n,
                                       llm_model=llm_model, embed_model=embed_model, config=config)
    retriever_sikepo_rek = lotr_sikepo(vector_store=vector_store_rek, top_n=top_n,
                                       llm_model=llm_model, embed_model=embed_model, config=config)

    # Reinitialize the chain with the new retrievers
    graph_chain = graph_rag_chain(llm_model, llm_model, graph=graph)
    chain = create_combined_answer_chain(
        llm_model=llm_model,
        graph_chain=graph_chain,
        retriever_ojk=retriever_ojk,
        retriever_bi=retriever_bi,
        retriever_sikepo_ketentuan=retriever_sikepo_ket,
        retriever_sikepo_rekam=retriever_sikepo_rek,
    )

    chain_history = create_chain_with_chat_history(
        final_chain=chain,
        chat_store=chat_store,
    )

    return JSONResponse(
        status_code=200,
        content={
            "message": "Model and parameters updated successfully",
            "model": model,
            "top_n": top_n
        }
    )

@app.get("/chat/{message}")
async def chat_endpoint(message: str):
    return StreamingResponse(print_answer_stream(message, chain=chain_history, user_id="user_id", conversation_id=CONVERSATION_ID), media_type="text/event-stream")

# @app.post("/chat/", response_model=ChatResponse)
# async def chat_input(request: ChatRequest, credentials: HTTPAuthorizationCredentials = Depends(security)):
#     if not request.user_input:
#         raise HTTPException(
#             status_code=400, detail="Please enter a valid input")

#     user_message = request.user_input
#     ai_response = ""

#     # Get the user ID from the Authorization header
#     user_id = credentials.credentials

#     response = get_response(
#         chain=chain_history,
#         question=user_message,
#         user_id=user_id,
#         conversation_id=CONVERSATION_ID,
#     )

#     ai_response = response['answer'].replace("\n", "<br>")

#     return JSONResponse(
#         status_code=200,
#         content={
#             "user_message": user_message,
#             "ai_response": ai_response
#         }
#     )
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9898)