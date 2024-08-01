from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from utils.config import get_config
from utils.models import ModelName, LLMModelName, EmbeddingModelName, get_model
from database.vector_store.vector_store import RedisIndexManager, ElasticIndexManager
from database.vector_store.neo4j_graph_store import Neo4jGraphStore
from retriever.retriever_ojk.retriever_ojk import get_retriever_ojk
from retriever.retriever_bi.retriever_bi import get_retriever_bi
from retriever.retriever_sikepo.lotr_sikepo import lotr_sikepo
from database.chat_store import RedisChatStore, ElasticChatStore
from chain.rag_chain import create_chain_with_chat_history, create_sequential_chain, create_combined_answer_chain, create_combined_context_chain
from retriever.retriever_sikepo.graph_cypher_retriever import graph_rag_chain
from chain.rag_chain import get_response

import warnings
warnings.filterwarnings("ignore")

# =========== CONFIG ===========
config = get_config()
# from rag import caller

USER_ID = 'xmriz'
CONVERSATION_ID = 'xmriz-2021-07-01-01'


# =========== MODEL ===========
model_name = ModelName.AZURE_OPENAI
llm_model, embed_model = get_model(model_name=model_name, config=config, llm_model_name=LLMModelName.GPT_35_TURBO, embedding_model_name=EmbeddingModelName.EMBEDDING_3_SMALL)

# =========== VECTOR STORE ===========
index_ojk = ElasticIndexManager(index_name='ojk', embed_model=embed_model, config=config)
vector_store_ojk = index_ojk.load_vector_index()

# index_bi = ElasticIndexManager(index_name='bi', embed_model=embed_model, config=config)
# vector_store_bi = index_bi.load_vector_index()
index_bi = ElasticIndexManager(index_name='sikepo-ketentuan-terkait', embed_model=embed_model, config=config)
vector_store_bi = index_bi.load_vector_index()

index_sikepo_ket = ElasticIndexManager(index_name='sikepo-ketentuan-terkait', embed_model=embed_model, config=config)
vector_store_ket = index_sikepo_ket.load_vector_index()

index_sikepo_rek = ElasticIndexManager(index_name='sikepo-rekam-jejak', embed_model=embed_model, config=config)
vector_store_rek = index_sikepo_rek.load_vector_index()

neo4j_sikepo = Neo4jGraphStore(config=config)
graph = neo4j_sikepo.get_graph()


# =========== RETRIEVER ===========
retriever_ojk = get_retriever_ojk(vector_store=vector_store_ojk, top_n=5, top_k=16, llm_model=llm_model, embed_model=embed_model, config=config)
retriever_bi = get_retriever_bi(vector_store=vector_store_bi, top_n=5, top_k=16, llm_model=llm_model, embed_model=embed_model, config=config)
retriever_sikepo_ket = lotr_sikepo(vector_store=vector_store_ket, llm_model=llm_model, embed_model=embed_model, config=config)
retriever_sikepo_rek = lotr_sikepo(vector_store=vector_store_rek, llm_model=llm_model, embed_model=embed_model, config=config)

# =========== CHAT STORE ===========
chat_store = ElasticChatStore(k=3, config=config)


# =========== CHAIN ===========
graph_chain = graph_rag_chain(llm_model, llm_model, graph=graph)

chain = create_combined_answer_chain(
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

# =========== MAIN ===========
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