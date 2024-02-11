import os
from typing import List

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Assuming you have a module named salesgpt with a class SalesGPTAPI
from salesgpt.salesgptapi import SalesGPTAPI

app = FastAPI()

GPT_MODEL = "gpt-3.5-turbo-0613"
# GPT_MODEL_16K = "gpt-3.5-turbo-16k-0613"


@app.get("/")
async def say_hello():
    return {"message": "Hello World"}


class MessageList(BaseModel):
    conversation_history: List[str]
    human_say: str


@app.post("/chat")
async def chat_with_sales_agent(req: MessageList):
    sales_api = SalesGPTAPI(
        config_path="examples/example_agent_setup.json", verbose=True
    )
    name, reply = sales_api.do(req.conversation_history, req.human_say)
    res = {"name": name, "say": reply}
    return res


def _set_env():
    # Ensure .env file exists and contains the necessary environment variables
    if os.path.isfile(".env"):
        with open(".env", "r") as f:
            env_file = f.readlines()
        envs_dict = {
            key.strip(): value.strip("\n")
            for key, value in (line.split("=") for line in env_file if line.strip())
        }
        os.environ["OPENAI_API_KEY"] = envs_dict.get("OPENAI_API_KEY", "")
    else:
        print(".env file not found, ensure OPENAI_API_KEY is set.")


if __name__ == "__main__":
    _set_env()
    # Use PORT environment variable if available, otherwise default to 8000 for local development
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
