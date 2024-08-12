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
from database.chat_store import ElasticChatStore
from chain.rag_chain import create_combined_answer_chain, create_chain_with_chat_history, print_answer_stream
from chain.chain_sikepo.graph_cypher_sikepo_chain import graph_rag_chain
from chain.rag_chain import get_response
from typing import AsyncGenerator
import time
from typing import Union
from fastapi import Query

import nest_asyncio
import warnings

nest_asyncio.apply()
warnings.filterwarnings("ignore")

# =========== CONFIG ===========
security = HTTPBearer()

config = get_config()

# =========== GLOBAL VARIABLES ===========
llm_model, embed_model = get_model(model_name=ModelName.AZURE_OPENAI, config=config,
                                   llm_model_name=LLMModelName.GPT_35_TURBO, embedding_model_name=EmbeddingModelName.EMBEDDING_3_SMALL)
top_k = 8

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
chat_store = ElasticChatStore(k=4, config=config)

# Initialize retrievers and chains with default model
retriever_ojk = get_retriever_ojk(vector_store=vector_store_ojk, top_k=top_k,
                                  llm_model=llm_model, embed_model=embed_model, config=config)
retriever_ojk_wo_self = get_retriever_ojk(vector_store=vector_store_ojk, top_k=top_k,
                                          llm_model=llm_model, embed_model=embed_model, config=config, with_self_query=False)
retriever_bi = get_retriever_bi(vector_store=vector_store_bi, top_k=top_k,
                                llm_model=llm_model, embed_model=embed_model, config=config)
retriever_sikepo_ket = lotr_sikepo(vector_store=vector_store_ket, top_k=top_k,
                                   llm_model=llm_model, embed_model=embed_model, config=config)
retriever_sikepo_ket_wo_self = lotr_sikepo(vector_store=vector_store_ket, top_k=top_k,
                                           llm_model=llm_model, embed_model=embed_model, config=config, with_self_query=False)
retriever_sikepo_rek = lotr_sikepo(vector_store=vector_store_rek, top_k=top_k,
                                   llm_model=llm_model, embed_model=embed_model, config=config)
retriever_sikepo_rek_wo_self = lotr_sikepo(vector_store=vector_store_rek, top_k=top_k,
                                           llm_model=llm_model, embed_model=embed_model, config=config, with_self_query=False)

graph_chain = graph_rag_chain(llm_model, llm_model, graph=graph)
chain = create_combined_answer_chain(
    llm_model=llm_model,
    graph_chain=graph_chain,
    retriever_ojk=retriever_ojk,
    retriever_bi=retriever_bi,
    retriever_sikepo_ketentuan=retriever_sikepo_ket,
    retriever_sikepo_rekam=retriever_sikepo_rek,
)
chain_wo_self = create_combined_answer_chain(
    llm_model=llm_model,
    graph_chain=graph_chain,
    retriever_ojk=retriever_ojk_wo_self,
    retriever_bi=retriever_bi,
    retriever_sikepo_ketentuan=retriever_sikepo_ket_wo_self,
    retriever_sikepo_rekam=retriever_sikepo_rek_wo_self,
)

chain_history = create_chain_with_chat_history(
    final_chain=chain,
    chat_store=chat_store,
)

chain_history_wo_self = create_chain_with_chat_history(
    final_chain=chain_wo_self,
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
    global llm_model, embed_model, retriever_ojk, retriever_ojk_wo_self, retriever_bi, retriever_sikepo_ket, retriever_sikepo_ket_wo_self, retriever_sikepo_rek, retriever_sikepo_rek_wo_self, chain, chain_wo_self, chain_history, chain_history_wo_self, top_k

    model = request.model
    print(f"Received model: {model}")

    if model == 'GPT_4O_MINI':
        llm_model, embed_model = get_model(model_name=ModelName.OPENAI, config=config,
                                           llm_model_name=LLMModelName.GPT_4O_MINI, embedding_model_name=EmbeddingModelName.EMBEDDING_3_SMALL)
        top_k = 12
    elif model == 'GPT_35_TURBO':
        llm_model, embed_model = get_model(model_name=ModelName.AZURE_OPENAI, config=config,
                                           llm_model_name=LLMModelName.GPT_35_TURBO, embedding_model_name=EmbeddingModelName.EMBEDDING_3_SMALL)
        top_k = 8
    else:
        raise HTTPException(status_code=400, detail="Invalid model specified")

    # Reinitialize retrievers with the new model and top_k
    retriever_ojk = get_retriever_ojk(vector_store=vector_store_ojk, top_k=top_k,
                                      llm_model=llm_model, embed_model=embed_model, config=config)
    retriever_ojk_wo_self = get_retriever_ojk(vector_store=vector_store_ojk, top_k=top_k,
                                              llm_model=llm_model, embed_model=embed_model, config=config, with_self_query=False)
    retriever_bi = get_retriever_bi(vector_store=vector_store_bi, top_k=top_k,
                                    llm_model=llm_model, embed_model=embed_model, config=config)
    retriever_sikepo_ket = lotr_sikepo(vector_store=vector_store_ket, top_k=top_k,
                                       llm_model=llm_model, embed_model=embed_model, config=config)
    retriever_sikepo_ket_wo_self = lotr_sikepo(vector_store=vector_store_ket, top_k=top_k,
                                               llm_model=llm_model, embed_model=embed_model, config=config, with_self_query=False)
    retriever_sikepo_rek = lotr_sikepo(vector_store=vector_store_rek, top_k=top_k,
                                       llm_model=llm_model, embed_model=embed_model, config=config)
    retriever_sikepo_rek_wo_self = lotr_sikepo(vector_store=vector_store_rek, top_k=top_k,
                                               llm_model=llm_model, embed_model=embed_model, config=config, with_self_query=False)

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
    chain_wo_self = create_combined_answer_chain(
        llm_model=llm_model,
        graph_chain=graph_chain,
        retriever_ojk=retriever_ojk_wo_self,
        retriever_bi=retriever_bi,
        retriever_sikepo_ketentuan=retriever_sikepo_ket_wo_self,
        retriever_sikepo_rekam=retriever_sikepo_rek_wo_self,
    )

    chain_history = create_chain_with_chat_history(
        final_chain=chain,
        chat_store=chat_store,
    )

    chain_history_wo_self = create_chain_with_chat_history(
        final_chain=chain_wo_self,
        chat_store=chat_store,
    )

    return JSONResponse(
        status_code=200,
        content={
            "message": "Model and parameters updated successfully",
            "model": model,
            "top_k": top_k
        }
    )


@app.get("/chat")
async def chat_endpoint(message: str, conv_id: str, credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Get the user ID from the Authorization header
    user_id = credentials.credentials
    try:
        response = StreamingResponse(print_answer_stream(
            message, chain=chain_history, user_id=user_id, conversation_id=conv_id), media_type="text/event-stream")
        return response
    except:
        response = StreamingResponse(print_answer_stream(
            message, chain=chain_history_wo_self, user_id=user_id, conversation_id=conv_id), media_type="text/event-stream")
        return response


@app.get("/fetch_conversations/")
async def fetch_conv(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Get the user ID from the Authorization header
    user_id = credentials.credentials
    session_ids_with_title = chat_store.get_conversation_ids_by_user_id(
        user_id)
    return JSONResponse(
        status_code=200,
        content=session_ids_with_title
    )


@app.get("/fetch_messages/{conversation_id}")
async def fetch_message(conversation_id: str, credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Get the user ID from the Authorization header
    user_id = credentials.credentials
    messages = chat_store.get_conversation(
        user_id=user_id, conversation_id=conversation_id)
    return JSONResponse(
        status_code=200,
        content=messages
    )


@app.put("/rename_conversation/{conversation_id}")
async def rename_conversation(
    conversation_id: str,
    new_title: str = Query(..., description="New title for the conversation"),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    user_id = credentials.credentials
    success = chat_store.rename_title(user_id, conversation_id, new_title)
    if success:
        return JSONResponse(status_code=200, content={"message": "Conversation renamed successfully"})
    else:
        raise HTTPException(status_code=404, detail="Conversation not found")


@app.delete("/delete_conversation/{conversation_id}")
async def delete_conversation(conversation_id: str, credentials: HTTPAuthorizationCredentials = Depends(security)):
    user_id = credentials.credentials
    success = chat_store.clear_session_history(user_id, conversation_id)
    if success:
        return JSONResponse(status_code=200, content={"message": "Conversation deleted successfully"})
    else:
        raise HTTPException(status_code=404, detail="Conversation not found")


@app.delete("/delete_all_conversations/")
async def delete_all_user_chats(credentials: HTTPAuthorizationCredentials = Depends(security)):
    user_id = credentials.credentials
    success = chat_store.clear_conversation_by_userid(user_id)
    if success:
        return JSONResponse(status_code=200, content={"message": "All conversations deleted successfully"})
    else:
        raise HTTPException(
            status_code=404, detail="No conversations found for the user")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9898)
