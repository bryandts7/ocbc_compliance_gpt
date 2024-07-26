from pydantic import BaseModel
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from random import randint
import queue
import asyncio
import nest_asyncio
from utils.config import get_config
from utils.models import ModelName, LLMModelName, EmbeddingModelName, get_model
from databases.vector_store import RedisIndexManager, PineconeIndexManager
from databases.neo4j_graph_store import Neo4jGraphStore
from retriever.retriever_ojk.retriever_ojk import get_retriever_ojk
from retriever.retriever_bi.retriever_bi import get_retriever_bi
from retriever.retriever_sikepo.lotr_sikepo import lotr_sikepo
from databases.chat_store import RedisChatStore, MongoDBChatStore
from chain.rag_chain import create_chain_with_chat_history, create_chain
from retriever.retriever_sikepo.graph_cypher_retriever import graph_rag_chain
from chain.rag_chain import get_response
nest_asyncio.apply()


from dotenv import load_dotenv
config = get_config()
# from rag import caller

USERNAME = "User"
AI_NAME = "Robot"
message_queue = queue.Queue()
sess_id = "W56PNA34XM"
USER_ID = 'xmriz'
CONVERSATION_ID = 'xmriz-2021-07-01-01'
model_name = ModelName.AZURE_OPENAI
llm_model, embed_model = get_model(model_name=model_name, config=config, llm_model_name=LLMModelName.GPT_35_TURBO, embedding_model_name=EmbeddingModelName.EMBEDDING_3_SMALL)

redis_ojk = RedisIndexManager(index_name='ojk', embed_model=embed_model, config=config, db_id=0)
vector_store_ojk = redis_ojk.load_vector_index()

redis_bi = RedisIndexManager(index_name='bi', embed_model=embed_model, config=config, db_id=0)
vector_store_bi = redis_bi.load_vector_index()

redis_sikepo_ket = RedisIndexManager(index_name='sikepo-ketentuan-terkait', embed_model=embed_model, config=config, db_id=0)
vector_store_ket = redis_sikepo_ket.load_vector_index()

redis_sikepo_rek = RedisIndexManager(index_name='sikepo-rekam-jejak', embed_model=embed_model, config=config, db_id=0)
vector_store_rek = redis_sikepo_rek.load_vector_index()

neo4j_sikepo = Neo4jGraphStore(config=config)
graph = neo4j_sikepo.get_graph()

retriever_ojk = get_retriever_ojk(vector_store=vector_store_ojk, top_n=6, top_k=16, llm_model=llm_model, embed_model=embed_model, config=config)
retriever_bi = get_retriever_bi(vector_store=vector_store_bi, top_n=6, top_k=16, llm_model=llm_model, embed_model=embed_model, config=config)
retriever_sikepo_ket = lotr_sikepo(vector_store=vector_store_ket, llm_model=llm_model, embed_model=embed_model, config=config)
retriever_sikepo_rek = lotr_sikepo(vector_store=vector_store_rek, llm_model=llm_model, embed_model=embed_model, config=config)

chat_store = RedisChatStore(k=3, config=config, db_id=1)

graph_chain = graph_rag_chain(llm_model, llm_model, graph=graph)

chain = create_chain(
    llm_model=llm_model,
    graph_chain=graph_chain,
    retriever_ojk=retriever_ojk,
    retriever_bi=retriever_sikepo_ket,
    retriever_sikepo_ketentuan=retriever_sikepo_ket,
    retriever_sikepo_rekam=retriever_sikepo_rek,
)

chain_history = create_chain_with_chat_history(
    final_chain=chain,
    chat_store=chat_store,
)

app = FastAPI()

class ChatRequest(BaseModel):
    user_input: str


class ChatResponse(BaseModel):
    user_message: str
    ai_response: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat/", response_model=ChatResponse)
async def chat_input(request: ChatRequest):
    if not request.user_input:
        raise HTTPException(
            status_code=400, detail="Please enter a valid input")

    user_message = request.user_input
    ai_response = ""

    response = get_response(
                    chain=chain_history,
                    question=user_message,
                    user_id=USER_ID,
                    conversation_id=CONVERSATION_ID
                )
    
    print(response)
    # ai_response = response['answer']['answer'].replace("\n", "<br>")
    ai_response = response['answer'].replace("\n", "<br>")

    # history.append({"user": user_message, "ai": ai_response})

    return JSONResponse(
        status_code=200,
        content={
            "user_message": user_message,
            "ai_response": ai_response
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9898)