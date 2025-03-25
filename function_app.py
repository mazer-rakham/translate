import azure.functions as func
import logging
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
import os
import json
from dotenv import load_dotenv

load_dotenv()

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# Load environment variables
model_id = "Llama-3.3-70B-Instruct"  # Update this to your Llama model ID if applicable
endpoint = ""
api_key = ""

if not model_id or not endpoint or not api_key:
    raise ValueError("Environment variables for Azure OpenAI are not set correctly.")

# Load the translation prompt
with open("prompts/Translate/skprompt.txt", "r") as prompt_file:
    prompt_template = prompt_file.read()

@app.route(route="sk_test", methods=["POST"])
def sk_test(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        req_body = req.get_json()
        user_input = req_body.get('input')
    except ValueError:
        return func.HttpResponse("Invalid input", status_code=400)

    if not user_input:
        return func.HttpResponse("Please provide input text", status_code=400)

    # Initialize the chat completions client
    client = ChatCompletionsClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))

    # Use the prompt template in your function logic
    system_message = SystemMessage(content="You are a helpful assistant.")
    user_message = UserMessage(content=prompt_template.replace("{{input}}", user_input))

    # Send the message and get the response
    response = client.complete(
        messages=[system_message, user_message],
        max_tokens=2048,
        temperature=0.8,
        top_p=0.1,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        model=model_id
    )

    # Extract the text content from the response
    response_text = response.choices[0].message.content if response.choices else "No response"

    # Return the translated text
    return func.HttpResponse(response_text, status_code=200)