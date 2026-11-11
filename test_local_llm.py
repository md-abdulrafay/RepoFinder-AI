from llama_cpp import Llama

llm = Llama(
    model_path="models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    n_ctx=2048,
    n_threads=4,
    n_batch=128,
    verbose=False
)

response = llm.create_chat_completion(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer briefly."
        },
        {
            "role": "user",
            "content": "Explain what a GitHub repository is in two sentences."
        }
    ],
    temperature=0.1,
    max_tokens=120
)

print(response["choices"][0]["message"]["content"])