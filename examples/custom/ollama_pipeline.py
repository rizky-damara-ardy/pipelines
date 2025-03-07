from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
import requests
import json


class Pipeline:
    def __init__(self):
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "ollama_pipeline"
        self.name = "Ollama Pipeline"
        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    def pipe(
            self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")
        OLLAMA_BASE_URL = "http://host.docker.internal:11434"
        MODEL = "llama3.1"
        if "user" in body:
            print("######################################")
            print(f'# User: {body["user"]["name"]} ({body["user"]["id"]})')
            print(f"# Message: {user_message}")
            print("######################################")
        try:
            r = requests.post(
                url=f"{OLLAMA_BASE_URL}/api/chat",
                json={**body, "model": MODEL},
                stream=True,
            )
            r.raise_for_status()

            # Initialize an empty string for collecting the response
            full_response = ""

            for line in r.iter_lines():
                if line:
                    # Parse the JSON line
                    json_line = line.decode('utf-8')
                    response_data = json.loads(json_line)

                    # Append the content to the full response
                    full_response += response_data['message']['content']

                    # Check if the message is done
                    if response_data.get("done"):
                        break

            return full_response
        except Exception as e:
            return f"Error: {e}"