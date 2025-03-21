from typing import List, Generator, Any
import requests
import json
import copy
import os

# LangChain and ChromaDB
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class Pipeline:
    def __init__(self):
        self.id = "coe_private_ai"
        self.name = "COE Private AI"

        # Ingest Document
        self.ingest_documents("./document.txt")

        # Load local embeddings model
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Initialize ChromaDB (local storage)
        self.vector_db = Chroma(persist_directory="./chroma_db", embedding_function=self.embedding_model)
        pass

    async def on_startup(self):
        print(f"on_startup:{__name__}")

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")

    def ingest_documents(self, file_path: str):
        """Load and embed documents into ChromaDB."""
        loader = TextLoader(file_path)
        documents = loader.load()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)

        # Add documents to ChromaDB
        self.vector_db.add_documents(docs)
        print(f"✅ {len(docs)} documents ingested into ChromaDB.")

    def retrieve_documents(self, query: str, top_k=3):
        """Retrieve relevant documents using similarity search."""
        docs = self.vector_db.similarity_search(query, k=top_k)
        return "\n".join([doc.page_content for doc in docs])

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Generator[str, None, None]:
        print(f"pipe:{__name__}")
        OLLAMA_BASE_URL = "http://host.docker.internal:11434"

        body_other = copy.deepcopy(body)
        body_cat = copy.deepcopy(body)
        body_llm = copy.deepcopy(body)

        if body['messages'][0]['content'].startswith('### Task'):
            # Direct request handling
            model = "qwen2.5:14b-instruct-q4_K_M"
            prompt = ""
            stream = False
            options_dict = {"temperature": 0.1, "context_length": 8192}

            yield from self.send_request_and_stream(user_message, body_other, OLLAMA_BASE_URL, model, prompt, stream, options_dict, False)
        else:
            # Determine request category
            category_generator = self.send_request_and_stream(
                user_message, body_cat, OLLAMA_BASE_URL, "qwen2.5:14b-instruct-q4_K_M",
                "Tentukan apakah permintaan ini dari pengguna merupakan meminta prediksi atau meminta pengetahuan atau meminta kode atau meminta gambar, jawabannya harus hanya 'prediksi' atau 'pengetahuan' atau 'kode' atau 'gambar' hanya itu.",
                False, {"temperature": 0.1, "context_length": 8192}, True
            )
            category = next(category_generator, "pengetahuan").strip().lower()

            if category == "pengetahuan":
                # Retrieve knowledge from ChromaDB
                retrieved_docs = self.retrieve_documents(user_message)
                system_prompt = f"Gunakan informasi berikut untuk menjawab pertanyaan pengguna:\n\n{retrieved_docs}\n\nJika dokumen tidak memiliki jawaban, jawab berdasarkan pengetahuan umum."

                model = "qwen2.5:14b-instruct-q4_K_M"
                options_dict = {"temperature": 0.2, "frequency_penalty": 0.2, "presence_penalty": 0.2, "num_ctx": 8192}
                yield f"knowledge-{model}: "

            elif category == "prediksi":
                model = "deepseek-r1:14b"
                system_prompt = "You are a prediction assistant, answer user questions."
                options_dict = {"temperature": 0.5, "frequency_penalty": 0.2, "presence_penalty": 0.2, "num_ctx": 8192}
                yield f"prediction-{model}: "

            elif category == "kode":
                model = "deepseek-coder-v2:16b"
                system_prompt = "You are a coding assistant, generate code for user queries."
                options_dict = {"num_ctx": 8192}
                yield f"code-{model}: "

            elif category == "gambar":
                model = "gemma3:27b"
                system_prompt = "You are an image assistant, generate images for user requests."
                options_dict = {"temperature": 0.5}
                yield f"image-{model}: "

            else:
                model = "qwen2.5:14b-instruct-q4_K_M"
                system_prompt = "You are a knowledge assistant, answer user queries."
                options_dict = {"num_ctx": 8192}
                yield f"general-{model}: "

            yield from self.send_request_and_stream(user_message, body_llm, OLLAMA_BASE_URL, model, system_prompt, False, options_dict, False)

    def send_request_and_stream(self, user_message: str, body: dict, base_url: str, model: str, system_prompt: str, stream: bool, option: dict, is_cat: bool) -> Generator[str | Any, Any, None]:
        OLLAMA_BASE_URL = base_url

        if system_prompt:
            if body['messages'][0]['role'] == 'system':
                body['messages'][0]['content'] = system_prompt if is_cat else system_prompt + ", " + body['messages'][0]['content']
            else:
                body['messages'].insert(0, {'role': 'system', 'content': system_prompt})

        payload = {**body, "model": model, "stream": stream, "options": option}

        try:
            r = requests.post(url=f"{OLLAMA_BASE_URL}/api/chat", json=self.adjust_json(payload))
            r.raise_for_status()

            for line in r.iter_lines():
                if line:
                    json_line = line.decode('utf-8')
                    response_data = json.loads(json_line)
                    content = response_data.get('message', {}).get('content', '')
                    if content:
                        yield content  # Stream the response content
        except Exception as e:
            yield f"Error: {e}"

    def adjust_json(self, owebui_json):
        """Adjusts JSON format for compatibility with the API."""
        new_json = {
            "model": owebui_json["model"],
            "stream": owebui_json["stream"],
            "messages": [],
        }
        if "options" in owebui_json:
            new_json["options"] = owebui_json["options"]

        def process_messages(messages):
            for message in messages:
                if message["role"] == "user":
                    if isinstance(message["content"], list):
                        for item in message["content"]:
                            if item["type"] == "text":
                                new_json["messages"].append({"role": "user", "content": item["text"]})
                            elif item["type"] == "image_url":
                                image_data = item["image_url"]["url"]
                                base64_string = image_data.split(",")[1]
                                new_json["messages"].append({"role": "user", "images": [base64_string]})
                    else:
                        new_json["messages"].append({"role": "user", "content": message["content"]})
                else:
                    new_json["messages"].append(message)

        if "messages" in owebui_json and isinstance(owebui_json["messages"], list):
            for top_message in owebui_json["messages"]:
                process_messages([top_message]) if isinstance(top_message, dict) and "messages" in top_message else process_messages([top_message])

        return new_json
