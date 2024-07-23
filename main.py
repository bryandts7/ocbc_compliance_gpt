from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import queue
from random import randint
from rag import caller
import asyncio

USERNAME = "User"
AI_NAME = "Robot"
message_queue = queue.Queue()
history = []
sess_id = "W56PNA34XM"

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    history.clear()
    return templates.TemplateResponse("index.html", {"request": request, "username": USERNAME, "ai_name": AI_NAME})

@app.post("/chat_submit/")
async def chat_input(user_input: str = Form(...)):
    if not user_input:
        ai_response = "Error: Please Enter a Valid Input"
        current_response_id = f"gptblock{randint(67, 999999)}"
        return templates.TemplateResponse("ai_response.html", {"ai_name": AI_NAME, "ai_response": ai_response, "hx_swap": False, "current_response_id": current_response_id })
    
    message_queue.put(user_input)
    return JSONResponse(content={"status": "Success"}, status_code=204)

@app.get('/stream')
async def stream():
    async def message_stream():
        global new_conversation

        while True:
            if not message_queue.empty():
                user_message = message_queue.get()
                current_response_id = f"gptblock{randint(67, 999999)}"
                hx_swap = False
                message = ""

                for word in caller(user_message, sess_id):
                    try:
                        message += word.replace("\n", "<br>")
                        ai_message = f"<p><strong>{AI_NAME}</strong> : {message}</p>"
                        res = f"""data: <li class="text-white p-4 m-2 shadow-md rounded bg-gray-800 text-sm" id="{current_response_id}" {"hx-swap-oob='true'" if hx_swap else ""}>{ai_message}</li>\n\n"""
                        hx_swap = True

                        print(f"{USERNAME}: {user_message}")
                        print(res)
                        yield res
        
                    except Exception as e:
                        print(e)
                        break
            await asyncio.sleep(0.1)
    
    return StreamingResponse(message_stream(), media_type='text/event-stream')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9898)
