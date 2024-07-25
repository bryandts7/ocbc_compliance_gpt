from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from random import randint
import queue
import asyncio
import nest_asyncio
from utils.config import get_config
from utils.models import ModelName, LLMModelName, EmbeddingModelName, get_model
from databases.vector_store import RedisIndexManager, PineconeIndexManager
from databases.neo4j_graph_store import Neo4jGraphStore
from retriever.retriever_ojk.retriever_ojk import get_retriever_ojk
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
llm_model, embed_model = get_model(model_name=model_name, config=config, llm_model_name=LLMModelName.GPT_35_TURBO, embedding_model_name=EmbeddingModelName.EMBEDDING_ADA)

pinecone_ojk = PineconeIndexManager(index_name='ojk', embed_model=embed_model, config=config)
vector_store_ojk = pinecone_ojk.load_vector_index()

pinecone_ket = PineconeIndexManager(index_name='ketentuan-terkait', embed_model=embed_model, config=config)
vector_store_ket = pinecone_ket.load_vector_index()

pinecone_rek = PineconeIndexManager(index_name='rekam-jejak', embed_model=embed_model, config=config)
vector_store_rek = pinecone_rek.load_vector_index()

neo4j_sikepo = Neo4jGraphStore(config=config)
graph = neo4j_sikepo.get_graph()

retriever_ojk = get_retriever_ojk(vector_store=vector_store_ojk, top_n=5, top_k=16, llm_model=llm_model, embed_model=embed_model, config=config)
retriever_sikepo_ket = lotr_sikepo(vector_store=vector_store_ket, llm_model=llm_model, embed_model=embed_model, config=config)
retriever_sikepo_rek = lotr_sikepo(vector_store=vector_store_rek, llm_model=llm_model, embed_model=embed_model, config=config)
chat_store = MongoDBChatStore(config=config, k=4)
graph_chain = graph_rag_chain(llm_model, llm_model, graph=graph)

chain = create_chain(retriever_ojk, retriever_sikepo_rek, retriever_sikepo_ket, graph_chain, llm_model)

chain_history = create_chain_with_chat_history(
    final_chain=chain,
    chat_store=chat_store,
)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "username": USERNAME, "ai_name": AI_NAME})

@app.post("/chat_submit/")
async def chat_input(user_input: str = Form(...)):
    if not user_input:
        ai_response = "Error: Please Enter a Valid Input"
        return templates.TemplateResponse("ai_response.html", {
            "ai_name": AI_NAME,
            "ai_response": ai_response,
            "hx_swap": False,
            "current_response_id": f"gptblock{randint(67, 999999)}"
        })
    
    message_queue.put(user_input)
    return JSONResponse(content={"status": "Success"}, status_code=204)

@app.get('/stream')
async def stream():
    async def message_stream():
        while True:
            if not message_queue.empty():
                user_message = message_queue.get()
                current_response_id = f"gptblock{randint(67, 999999)}"

                # Display "thinking..." message
                yield f"""data: <li class="text-white p-4 m-2 shadow-md rounded bg-gray-800 text-sm" id="{current_response_id}">thinking...</li>\n\n"""

                try:
                    # Directly get the response from the caller function
                    response = get_response(
                        chain=chain_history,
                        question=user_message,
                        user_id=USER_ID,
                        conversation_id=CONVERSATION_ID
                    )
                    message = response['answer']['answer'].replace("\n", "<br>")
                    # message = caller(user_message, sess_id).replace("\n", "<br>")
                    ai_message = f"<p><strong>{AI_NAME}</strong> : {message}</p>"

                    # Display the AI's message, replacing the "thinking..." message
                    yield f"""data: <li class="text-white p-4 m-2 shadow-md rounded bg-gray-800 text-sm" id="{current_response_id}" hx-swap-oob='true'>{ai_message}</li>\n\n"""
                except Exception as e:
                    print(e)
                    break
            await asyncio.sleep(0.1)

    return StreamingResponse(message_stream(), media_type='text/event-stream')
