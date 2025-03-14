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

        #for define category
        model = "qwen2.5:14b-instruct-q4_K_M"
        prompt = 'Define this prompt from user is ask prediction or ask knowledge or ask code, answer must only "prediction" or "knowledge" or "code" just it'
        stream = False
        options_str = '{"temperature": 0.1,"context_length": 8192}'
        options_dict = json.loads(options_str)

        category_generator = self.send_request_and_stream(user_message, body, OLLAMA_BASE_URL, model,prompt, stream, options_dict)

        # Retrieve the first result from the generator
        category = next(category_generator)

        category_lower = category.lower() if isinstance(category, str) else category

        if "knowledge" in category_lower:
            model = "qwen2.5:14b-instruct-q4_K_M"
            prompt = 'You are knowledge base assistant, answer user asking'
            stream = True
            options_str = '{"temperature": 0.2,"frequency_penalty": 0.2, "presence_penalty": 0.2, "num_ctx": 8192}'
            options_dict = json.loads(options_str)

            yield "knowledge-"+model+": "
        elif "prediction" in category_lower:
            model = "deepseek-r1:14b"
            prompt = 'You are prediction base assistant, answer user asking'
            stream = True
            options_str = '{"temperature": 0.5,"frequency_penalty": 0.2, "presence_penalty": 0.2, "num_ctx": 8192}'
            options_dict = json.loads(options_str)

            yield "prediction-"+model+": "
        elif "code" in category_lower:
            model = "deepseek-coder-v2:16b"
            prompt = 'You are code base assistant, answer user asking'
            stream = True
            options_str = '{"num_ctx": 8192}'
            options_dict = json.loads(options_str)

            yield "code-"+model+": "
        else:
            model = "qwen2.5:14b-instruct-q4_K_M"
            prompt = 'You are knowledge base assistant, answer user asking'
            stream = True
            options_str = '{"num_ctx": 8192}'
            options_dict = json.loads(options_str)

            yield "general-"+model+": "
        yield from self.send_request_and_stream(user_message, body, OLLAMA_BASE_URL, model,prompt, stream, options_dict)

    def send_request_and_stream(self, user_message: str, body: dict, base_url:str, model: str, system_prompt:str, stream: bool, option: dict) -> \
    Generator[str | Any, Any, None]:
        OLLAMA_BASE_URL = base_url

        body['messages'][0]['content'] =system_prompt
        payload = {**body, "model": model, "stream": stream, "options": option}

        try:
            print("XXX JSON XXX:",payload)
            print("YYY JSON YYY:",self.adjust_json(payload),)
            r = requests.post(
                url=f"{OLLAMA_BASE_URL}/api/chat",
                json=self.adjust_json(payload),
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

    def adjust_json(self,owebui_json):
        # Preparing the new structure
        new_json = {
            "model": owebui_json["model"],
            "stream": owebui_json["stream"],
            "user": owebui_json["user"],
            "messages": []
        }
        if "options" in owebui_json:
            new_json["options"] = owebui_json["options"]

        # Function to recursively process messages
        def process_messages(messages):
            for message in messages:
                if message["role"] == "user":
                    # Handle content as a list of structured items
                    if isinstance(message["content"], list):
                        for item in message["content"]:
                            if item["type"] == "text":
                                text_message = {
                                    "role": "user",
                                    "content": item["text"]
                                }
                                new_json["messages"].append(text_message)
                            elif item["type"] == "image_url":
                                image_data = item["image_url"]["url"]
                                base64_string = image_data.split(",")[1]
                                image_message = {
                                    "role": "user",
                                    "images": [base64_string]
                                }
                                new_json["messages"].append(image_message)
                    else:
                        # If it's a plain string
                        user_message = {
                            "role": "user",
                            "content": message["content"]
                        }
                        new_json["messages"].append(user_message)
                else:
                    # Append other roles (system, assistant) as is
                    new_json["messages"].append(message)

        # Process top-level messages
        if "messages" in owebui_json and isinstance(owebui_json["messages"], list):
            # If top-level messages are provided, process them
            for top_message in owebui_json["messages"]:
                if isinstance(top_message, dict) and "messages" in top_message:
                    process_messages(top_message["messages"])
                else:
                    process_messages([top_message])  # Handle messages that are not nested

        return new_json
