from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from random import randint
import queue
import asyncio

from rag import caller

USERNAME = "User"
AI_NAME = "Robot"
message_queue = queue.Queue()
sess_id = "W56PNA34XM"

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
                    message = caller(user_message, sess_id).replace("\n", "<br>")
                    ai_message = f"<p><strong>{AI_NAME}</strong> : {message}</p>"

                    # Display the AI's message, replacing the "thinking..." message
                    yield f"""data: <li class="text-white p-4 m-2 shadow-md rounded bg-gray-800 text-sm" id="{current_response_id}" hx-swap-oob='true'>{ai_message}</li>\n\n"""
                except Exception as e:
                    print(e)
                    break
            await asyncio.sleep(0.1)

    return StreamingResponse(message_stream(), media_type='text/event-stream')

