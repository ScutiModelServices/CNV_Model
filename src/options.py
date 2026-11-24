"""The options the admin page can change. Each can still start from its environment variable; a saved value wins."""

from settings import Setting

SETTINGS = [
    Setting("max_inflight", "Calls allowed to wait for the model", "int", 0, "SERVICE_MAX_INFLIGHT", 0, 10000,
            help="Per replica; more get 429. 0 = 4"),
    Setting("cache_size", "Answers kept per replica", "int", 256, "SERVICE_CACHE_SIZE", 0, 100000,
            help="The last answers for the same request, sent again without generating. Only temperature 0 requests are kept. 0 = off"),
    Setting("ollama_url", "Ollama server address", "text", "http://127.0.0.1:11434", "SERVICE_OLLAMA_URL", blank=False,
            help="Where the ollama models run, for example http://host.docker.internal:11434. Applies within a second; loaded models are dropped"),
    Setting("models_dir", "Folder of GGUF files", "text", "models", "SERVICE_MODELS_DIR", blank=False,
            help="Where the llama.cpp models' GGUF files are downloaded to and read from. In Docker it is /models; set the host folder with SERVICE_MODELS_PATH in compose. Ollama models are stored by the Ollama server, not here. Applies within a second; loaded models are dropped"),
    Setting("ollama_parallel", "Generations at once per replica", "int", 2, "SERVICE_OLLAMA_PARALLEL", 1, 64,
            help="Set it to the OLLAMA_NUM_PARALLEL of your Ollama. Applies within a second"),
    Setting("thinking", "Thinking", "text", "off", "SERVICE_THINKING", blank=False,
            help="off or on. on lets the model think before it answers: slower, and it needs a model that supports thinking. Changing this setting uses a separate answer cache key"),
    Setting("hide_reasoning", "Hide reasoning written into the answer", "bool", True, "SERVICE_HIDE_REASONING",
            help="Drops everything before the last </think>. A streamed reply is then sent when complete, in one piece"),
]
