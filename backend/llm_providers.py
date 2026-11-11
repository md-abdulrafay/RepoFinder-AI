import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from llama_cpp import Llama

load_dotenv()

_gemini_models = {}
_local_llm = None


def get_gemini(model_name):
    if model_name not in _gemini_models:
        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError("GOOGLE_API_KEY is missing")

        _gemini_models[model_name] = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0,
            timeout=20,
            max_retries=1
        )

    return _gemini_models[model_name]


def get_local_llm():
    global _local_llm

    if _local_llm is None:
        model_path = os.getenv(
            "LOCAL_MODEL_PATH",
            "models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Local model not found: {model_path}"
            )

        _local_llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            n_batch=128,
            verbose=False
        )

    return _local_llm


def generate_with_gemini(prompt, model_name):
    response = get_gemini(model_name).invoke(prompt)
    return response.content


def generate_with_local(prompt):
    response = get_local_llm().create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a query understanding assistant. "
                    "Follow the user's instructions and return JSON only."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1,
        max_tokens=300
    )

    return response["choices"][0]["message"]["content"]