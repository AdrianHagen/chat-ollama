"""Frontend utility functions for the Streamlit chat application.

This module provides helper functions for the frontend application,
including chat title generation and other UI utilities.
"""

import ollama


def generate_chat_title(first_message: str, model: str) -> str:
    """Generate a 3-word title for a chat based on the first message.
    
    Queries the Ollama LLM to create a concise 3-word summarization
    of the chat conversation based on the user's first message.
    
    Args:
        first_message: The first user message in the chat conversation.
        model: The name of the Ollama model to use for generation.
        
    Returns:
        str: A 3-word title for the chat. Returns a default title if
            generation fails or the message is empty.
            
    Examples:
        >>> generate_chat_title("How do I sort a list in Python?", "gemma3:12b")
        "Python List Sorting"
        >>> generate_chat_title("What is machine learning?", "llama2")
        "Machine Learning Basics"
    """
    if not first_message or not first_message.strip():
        return "New Chat"
    
    prompt_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that creates concise chat titles for a chat history sidebar. "
                       "Users will see these titles as buttons to identify and select their past conversations. "
                       "The title should clearly describe the main topic or question. "
                       "Respond with EXACTLY 3 words, nothing else. No punctuation, no explanation, no extra text."
        },
        {
            "role": "user",
            "content": f"Create a clear 3-word title that describes this conversation topic: \"{first_message}\"\n\n"
                       f"The title will be displayed as a button label in a chat history list. "
                       f"Make it descriptive and easy to understand at a glance."
        }
    ]
    
    try:
        response = ollama.chat(
            model=model,
            messages=prompt_messages,
            stream=False
        )
        
        title = response['message']['content'].strip()
        
        # Clean up the response - remove extra punctuation and limit to 3 words
        words = title.replace('.', '').replace('!', '').replace('?', '').split()
        
        # Take only the first 3 words
        if len(words) >= 3:
            return ' '.join(words[:3])
        elif len(words) > 0:
            return ' '.join(words)
        else:
            return "New Chat"
            
    except Exception as e:
        print(f"Error generating chat title: {e}")
        # Fallback: create a simple title from the first few words of the message
        words = first_message.split()[:3]
        if words:
            return ' '.join(words)
        return "New Chat"
