import warnings
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json 
from utils.config import get_config
from utils.models import ModelName, LLMModelName, EmbeddingModelName, get_model
from database.vector_store.vector_store import RedisIndexManager, ElasticIndexManager
from database.vector_store.neo4j_graph_store import Neo4jGraphStore
from retriever.retriever_ojk.retriever_ojk import get_retriever_ojk
from retriever.retriever_bi.retriever_bi import get_retriever_bi
from retriever.retriever_sikepo.lotr_sikepo import lotr_sikepo
from database.chat_store import RedisChatStore, ElasticChatStore
from chain.rag_chain import create_chain_with_chat_history, create_sequential_chain, create_combined_answer_chain, create_combined_context_chain
from chain.chain_sikepo.graph_cypher_sikepo_chain import graph_rag_chain
from chain.rag_chain import get_response

import nest_asyncio
nest_asyncio.apply()

warnings.filterwarnings("ignore")

# =========== CONFIG ===========
config = get_config()
# from rag import caller

USER_ID = 'xmriz'
CONVERSATION_ID = 'xmriz-2021-07-01-01'


# =========== MODEL ===========
model_name = ModelName.OPENAI
llm_model, embed_model = get_model(model_name=model_name, config=config,
                                   llm_model_name=LLMModelName.GPT_4O_MINI, embedding_model_name=EmbeddingModelName.EMBEDDING_3_SMALL)

# =========== VECTOR STORE ===========
index_ojk = ElasticIndexManager(
    index_name='ojk', embed_model=embed_model, config=config)
vector_store_ojk = index_ojk.load_vector_index()

index_bi = ElasticIndexManager(
    index_name='bi', embed_model=embed_model, config=config)
vector_store_bi = index_bi.load_vector_index()

index_sikepo_ket = ElasticIndexManager(
    index_name='sikepo-ketentuan-terkait', embed_model=embed_model, config=config)
vector_store_ket = index_sikepo_ket.load_vector_index()

index_sikepo_rek = ElasticIndexManager(
    index_name='sikepo-rekam-jejak', embed_model=embed_model, config=config)
vector_store_rek = index_sikepo_rek.load_vector_index()

neo4j_sikepo = Neo4jGraphStore(config=config)
graph = neo4j_sikepo.get_graph()


# =========== RETRIEVER ===========
retriever_ojk = get_retriever_ojk(vector_store=vector_store_ojk, top_n=7,
                                  llm_model=llm_model, embed_model=embed_model, config=config)
retriever_bi = get_retriever_bi(vector_store=vector_store_bi, top_n=7,
                                llm_model=llm_model, embed_model=embed_model, config=config)
retriever_sikepo_ket = lotr_sikepo(
    vector_store=vector_store_ket, top_n=7, llm_model=llm_model, embed_model=embed_model, config=config)
retriever_sikepo_rek = lotr_sikepo(
    vector_store=vector_store_rek, top_n=7, llm_model=llm_model, embed_model=embed_model, config=config)

# =========== CHAT STORE ===========
chat_store = ElasticChatStore(k=3, config=config)


# =========== CHAIN ===========
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


@app.post("/chat/")
async def chat_input(request: ChatRequest):
    if not request.user_input:
        raise HTTPException(
            status_code=400, detail="Please enter a valid input"
        )

    user_message = request.user_input

    async def response_generator():
        ai_response = ""
        yield '{"user_message": ' + json.dumps(user_message) + ', "ai_response": "'

        buffer = ""
        async for chunk in get_response(
            question=user_message,
            chain=chain_history,
            user_id=USER_ID,
            conversation_id=CONVERSATION_ID
        ):
            if isinstance(chunk, dict) and 'text' in chunk:
                buffer += chunk['text']
            else:
                buffer += str(chunk)

            # Only yield when we have a full sentence or a significant chunk
            if '.' in buffer or len(buffer) > 100:  # Adjust threshold as needed
                ai_response += buffer.strip().replace("\n", "<br>")
                yield ai_response
                buffer = ""  # Clear the buffer after yielding

        # Yield any remaining text in the buffer
        if buffer:
            ai_response += buffer.strip().replace("\n", "<br>")
            yield ai_response

        yield '"}'

    return StreamingResponse(response_generator(), media_type="application/json")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9898)


