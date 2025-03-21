from typing import List, Union, Generator, Iterator, Any
from schemas import OpenAIChatMessage
import requests
import json
import copy


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

        body_other = copy.deepcopy(body)
        body_cat = copy.deepcopy(body)
        body_llm = copy.deepcopy(body)

        print(f'---body_ori_before---:{body}')

        if body['messages'][0]['content'].startswith('### Task'):
            # for define category
            model = "qwen2.5:14b-instruct-q4_K_M"
            prompt = ""
            stream = False
            options_str = '{"temperature": 0.1,"context_length": 8192}'
            options_dict = json.loads(options_str)

            yield from self.send_request_and_stream(user_message, body_other, OLLAMA_BASE_URL, model, prompt,
                                           stream, options_dict,False)
        else:
            #for define category
            model = "qwen2.5:14b-instruct-q4_K_M"
            prompt = "Tentukan apakah permintaan ini dari pengguna merupakan meminta prediksi atau meminta pengetahuan atau meminta kode atau meminta gambar, jawabannya harus hanya 'prediksi' atau 'pengetahuan' atau 'kode' atau 'gambar' hanya itu."
            stream = False
            options_str = '{"temperature": 0.1,"context_length": 8192}'
            options_dict = json.loads(options_str)

            category_generator = self.send_request_and_stream(user_message, body_cat, OLLAMA_BASE_URL, model,prompt, stream, options_dict,True)

            # Retrieve the first result from the generator
            category = next(category_generator)
            category_lower = category.lower() if isinstance(category, str) else category

            if "pengetahuan" in category_lower:
                model = "qwen2.5:14b-instruct-q4_K_M"
                prompt = 'You are knowledge base assistant, answer user asking'
                stream = False
                options_str = '{"temperature": 0.2,"frequency_penalty": 0.2, "presence_penalty": 0.2, "num_ctx": 8192}'
                options_dict = json.loads(options_str)

                yield "knowledge-"+model+": "
            elif "prediksi" in category_lower:
                model = "deepseek-r1:14b"
                prompt = 'You are prediction base assistant, answer user asking'
                stream = False
                options_str = '{"temperature": 0.5,"frequency_penalty": 0.2, "presence_penalty": 0.2, "num_ctx": 8192}'
                options_dict = json.loads(options_str)

                yield "prediction-"+model+": "
            elif "kode" in category_lower:
                model = "deepseek-coder-v2:16b"
                prompt = 'You are code base assistant, answer user asking'
                stream = False
                options_str = '{"num_ctx": 8192}'
                options_dict = json.loads(options_str)

                yield "code-"+model+": "
            elif "gambar" in category_lower:
                model = "gemma3:27b"
                prompt = 'You are image base assistant, answer user asking'
                stream = False
                options_str = '{"temperature": 0.5}'
                options_dict = json.loads(options_str)

                yield "image-" + model + ": "
            else:
                model = "qwen2.5:14b-instruct-q4_K_M"
                prompt = 'You are knowledge base assistant, answer user asking'
                stream = False
                options_str = '{"num_ctx": 8192}'
                options_dict = json.loads(options_str)

                yield "general-"+model+": "
            yield from self.send_request_and_stream(user_message, body_llm, OLLAMA_BASE_URL, model,prompt, stream, options_dict, False)

    def send_request_and_stream(self, user_message: str, body: dict, base_url:str, model: str, system_prompt:str, stream: bool, option: dict, is_cat: bool) -> \
    Generator[str | Any, Any, None]:
        OLLAMA_BASE_URL = base_url

        print(f'000model000:{model}')
        print(f'111body_ori111:{body}')
        print(f'222system_prompt222:{system_prompt}')

        if system_prompt != "":
            if body['messages'][0]['role'] == 'system':
                if body['messages'][0]['content'].startswith(' ### Task'):
                    if is_cat:
                        body['messages'][0]['content'] = system_prompt
                    else:
                        body['messages'][0]['content'] = system_prompt+", "+body['messages'][0]['content']
                else:
                    body['messages'][0]['content'] = system_prompt
            else:
                body['messages'].insert(0, {'role': 'system', 'content': system_prompt})

        payload = {**body, "model": model, "stream": stream, "options": option}

        print(f'333body_modif333:{body}')

        try:
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
                        print(f'444content444:{content}')
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
