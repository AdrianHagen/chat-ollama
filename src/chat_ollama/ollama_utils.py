from typing import Dict, List

import ollama

def get_model(model_name: str):
    """Ensure the specified Ollama model is available locally.

    Checks whether the model identified by ``model_name`` is present locally by
    calling ``ollama.show``. If the model is not available, attempts to pull
    it using ``ollama.pull`` and prints status messages to stdout.

    Args:
        model_name (str): Name of the Ollama model to check or pull.

    Returns:
        None

    Notes:
        This function handles exceptions from the Ollama client by printing
        informational messages. It does not raise on failure to pull; callers
        that need stricter behavior should check for the model separately.
    """
    try:
        ollama.show(model_name)
        print(f"Model {model_name} already pulled.")
        return
    except Exception:
        print(f"Model {model_name} not found locally.")

    try:
        ollama.pull(model_name)
        print(f"Pulling {model_name} from ollama.")
    except Exception:
        print(f"Model {model_name} not found on ollama, please use a different model")


def chat_model(model_name: str, messages: List[Dict]):
    """Stream chat responses from an Ollama model.

    Ensures the requested model is available locally and then streams
    responses from the Ollama client. This function is a generator that yields
    each chunk of content as it is received from the streaming API.

    Args:
        model_name (str): Model identifier to use for chat (for example
            ``"gemma3:12b"``).
        messages (List[Dict]): A list of message dictionaries, typically
            containing keys such as ``"role"`` and ``"content"``.

    Yields:
        str: Chunks of model output content as they are streamed.

    Raises:
        Exception: Any exceptions raised by the Ollama client while streaming
            are propagated to the caller.
    """
    get_model(model_name=model_name)
    res = ollama.chat(model=model_name, messages=messages, stream=True)
    for part in res:
        print(part.message.content, end="", flush=True)
        yield part.message.content