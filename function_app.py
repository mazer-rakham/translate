import azure.functions as func
import logging
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)
import os
import json
from dotenv import load_dotenv

load_dotenv()

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# Load environment variables
model_id = os.getenv("AZURE_OPENAI_MODEL_NAME")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")

if not model_id or not endpoint or not api_key:
    raise ValueError("Environment variables for Azure OpenAI are not set correctly.")

@app.route(route="sk_test", methods=["POST"])
async def sk_test(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        req_body = req.get_json()
        user_input = req_body.get('input')
    except ValueError:
        return func.HttpResponse("Invalid input", status_code=400)

    if not user_input:
        return func.HttpResponse("Please provide input text", status_code=400)

    # Initialize the kernel
    kernel = Kernel()

    # Add Azure OpenAI chat completion
    chat_completion = AzureChatCompletion(
        deployment_name=model_id,
        api_key=api_key,
        base_url=endpoint,
    )
    kernel.add_service(chat_completion)

    # Load the skill configuration
    with open("skills/Translate/config.json", "r") as config_file:
        skill_config = json.load(config_file)

    # Load the translation prompt
    with open("skills/Translate/skprompt.txt", "r") as prompt_file:
        prompt_template = prompt_file.read()

    # Create a history of the conversation
    history = ChatHistory()
    history.add_user_message(prompt_template.replace("{{input}}", user_input))

    # Set execution settings based on config
    execution_settings = AzureChatPromptExecutionSettings(
        max_tokens=skill_config["execution_settings"]["default"]["max_tokens"],
        temperature=skill_config["execution_settings"]["default"]["temperature"]
    )

    # Get the response from the AI
    result = await chat_completion.get_chat_message_content(
        chat_history=history,
        settings=execution_settings,
        kernel=kernel,
    )

    # Extract the text content from the result
    response_text = result.content if hasattr(result, 'content') else str(result)

    # Return the translated text
    return func.HttpResponse(response_text, status_code=200)