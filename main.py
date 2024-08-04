from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from gradientai import Gradient
from supabase import create_client, Client
import os
import uvicorn
import json


# Supabase setup
SUPABASE_URL = "https://qbhmbdwocbvnhsampfmm.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFiaG1iZHdvY2J2bmhzYW1wZm1tIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjExMjM4NzIsImV4cCI6MjAzNjY5OTg3Mn0.-bF1bcBAkawL5eXeN5gjJuP4FdAqL4W2Cvtk4_g2aSE"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# GradientAI setup
os.environ['GRADIENT_ACCESS_TOKEN'] = "6w1HNB22pCDR9axSYkRFp8NWFe5zjXRT"
os.environ['GRADIENT_WORKSPACE_ID'] = "81a704a0-fa72-484c-a188-aaf4f4083be8_workspace"
MODEL_ADAPTERS_FILE = 'model_adapters.json'

app = FastAPI()

origins = [
    "http://127.0.0.1:5500",  # Your frontend origin
    "http://localhost:5500",
    "https://essay-llm-finetuner.vercel.app",   # Sometimes localhost is used
    "http://essay-llm-finetuner.vercel.app",
    "http://127.0.0.1:8000",
    "https://finetune.krishshroff.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Number(BaseModel):
    value: int


class GenerateRequest(BaseModel):
    user_id: int
    prompt: str


def fetch_user_data(user_id):
    response = supabase.table("essay-data").select("*").eq("userid", user_id).execute()
    # if response.status_code != 200:
    #     raise HTTPException(status_code=response.status_code, detail=response.json().get('message', 'Error fetching data from Supabase'))
    return response.data

model_adapters = {}
def save_model_adapters():
    with open(MODEL_ADAPTERS_FILE, 'w') as f:
        json.dump(model_adapters, f)
# @app.post("/fine-tune")
# async def process_number(number: Number):
#     print(f"Received number: {number.value}")
#     user_data = fetch_user_data(number.value)
#     print(user_data)
#     # if not user_data::
#     #     raise HTTPException(status_code=404, detail="No data found for the given user ID")
#     return {"received_number": number.value}
@app.post("/fine-tune")
async def fine_tune(number: Number):
    try:
        user_data = fetch_user_data(number.value)
        if not user_data:
            raise HTTPException(status_code=404, detail="No data found for the given user ID")

        with Gradient() as gradient:
            base_model = gradient.get_base_model(base_model_slug="nous-hermes2")
            new_model_adapter = base_model.create_model_adapter(name=f"model_for_{number.value}")
            adapter_id = new_model_adapter.id
            model_adapters[f"model_for_{number.value}"] = adapter_id

            save_model_adapters()

            sample_query = "### Instruction: Write an 300 word essay about AI impact on the world? \n\n### Response:"
            # print(f"Asking: {sample_query}")

            # # before fine-tuning
            # completion = new_model_adapter.complete(query=sample_query, max_generated_token_count=100).generated_output
            # print(f"Generated (before fine-tune): {completion}")

            samples = [{"inputs": f"### Instruction: {record['prompt']} \n\n### Response: {record['content']}"} for record in user_data]
            num_epochs = 3
            for epoch in range(num_epochs):
                print(f"Fine-tuning the model, iteration {epoch + 1}")
                new_model_adapter.fine_tune(samples=samples)

            # completion_after = new_model_adapter.complete(query=sample_query, max_generated_token_count=100).generated_output
            # print(f"Generated (after fine-tune): {completion_after}")
           
            return {
                "message": "Model fine-tuned successfully with model number " + str(number.value),
            }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    

@app.post("/generate")
async def generate(request: GenerateRequest):
    try:
        model_name = f"model_for_{request.user_id}"
        print(f"Looking for model adapter with name: {model_name}")
        
        with Gradient() as gradient:
            model_name = f"model_for_{request.user_id}"
            print(f"Looking for model adapter with name: {model_name}")

            # Load model adapters dictionary from the file
            if os.path.exists(MODEL_ADAPTERS_FILE):
                with open(MODEL_ADAPTERS_FILE, 'r') as f:
                    model_adapters = json.load(f)
            else:
                raise HTTPException(status_code=404, detail="Model adapters file not found")

            # Retrieve the adapter ID
            model_adapter_id = model_adapters.get(model_name)
            
            # Access the model adapter using the found ID
            model_adapter = gradient.get_model_adapter(model_adapter_id=model_adapter_id)
            completion = model_adapter.complete(query=request.prompt, max_generated_token_count=500).generated_output

        return {"generated_text": completion}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")