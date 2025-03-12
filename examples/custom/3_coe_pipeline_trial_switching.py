from typing import List, Union, Generator, Iterator, Any
from schemas import OpenAIChatMessage
import requests
import json


class Pipeline:
    def __init__(self):
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        self.id = "coe_private_ai"
        self.name = "COE Private AI"
        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Generator[str, None, None]:
        print(f"pipe:{__name__}")
        OLLAMA_BASE_URL = "http://host.docker.internal:11434"

        category_generator = self.send_request_and_stream(user_message, body, OLLAMA_BASE_URL, "llama3.1:latest",
                                                          'Define this prompt from user is ask prediction or ask knowledge or ask code, answer must only "prediction" or "knowledge" or "code" just it',
                                                          False)

        # Retrieve the first result from the generator
        category = next(category_generator)

        category_lower = category.lower() if isinstance(category, str) else category

        if "knowledge" in category_lower:
            model = "llama3.1:latest"
            prompt = 'You are knowledge base assistant, answer user asking'
            stream = True
            yield "knowledge-"+model+": "
        elif "prediction" in category_lower:
            model = "deepseek-r1:14b"
            prompt = 'You are prediction base assistant, answer user asking'
            stream = True
            yield "prediction-"+model+": "
        elif "code" in category_lower:
            model = "deepseek-coder-v2:16b"
            prompt = 'You are code base assistant, answer user asking'
            stream = True
            yield "code-"+model+": "
        else:
            model = "llama3.1:latest"
            prompt = 'You are knowledge base assistant, answer user asking'
            stream = True
            yield "general-"+model+": "
        yield from self.send_request_and_stream(user_message, body, OLLAMA_BASE_URL, model,
                                                prompt,
                                                stream)

    def send_request_and_stream(self, user_message: str, body: dict, base_url:str, model: str, system_prompt:str, stream: bool) -> \
    Generator[str | Any, Any, None]:
        OLLAMA_BASE_URL = base_url

        body['messages'][0]['content'] =system_prompt

        try:
            r = requests.post(
                url=f"{OLLAMA_BASE_URL}/api/chat",
                json={**body, "model": model},
                stream=stream,
            )
            r.raise_for_status()

            for line in r.iter_lines():
                if line:
                    json_line = line.decode('utf-8')
                    response_data = json.loads(json_line)
                    content = response_data.get('message', {}).get('content', '')
                    if content:
                        yield content  # Stream the content
        except Exception as e:
            yield f"Error: {e}"