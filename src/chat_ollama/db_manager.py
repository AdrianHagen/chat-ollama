"""Database manager for chat history.

This module provides functions to manage chat conversations and messages
in an SQLite database.
"""

import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Any


def get_db_path() -> Path:
    """Get the path to the SQLite database file.
    
    Returns:
        Path: The path to the database file.
    """
    db_dir = Path(__file__).parent.parent.parent / "db"
    db_dir.mkdir(exist_ok=True)
    return db_dir / "chats.db"


def _get_connection() -> sqlite3.Connection:
    """Get a database connection with foreign keys enabled.
    
    Returns:
        sqlite3.Connection: Database connection with foreign keys enabled.
    """
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_database() -> None:
    """Initialize the database with required tables.
    
    Creates the following tables:
    - chats: Stores chat sessions with metadata
    - messages: Stores individual messages within chats
    """
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    cursor = conn.cursor()
    
    # Create chats table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            model TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create messages table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chat_id) REFERENCES chats (id) ON DELETE CASCADE
        )
    """)
    
    # Create index for faster queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_chat_id 
        ON messages (chat_id)
    """)
    
    conn.commit()
    conn.close()


def create_chat(title: str, model: str) -> int:
    """Create a new chat session.
    
    Args:
        title: The title of the chat.
        model: The model name used for this chat.
        
    Returns:
        int: The ID of the newly created chat.
    """
    conn = _get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "INSERT INTO chats (title, model) VALUES (?, ?)",
        (title, model)
    )
    
    chat_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return chat_id


def get_all_chats() -> List[Dict[str, Any]]:
    """Retrieve all chat sessions.
    
    Returns:
        List[Dict[str, Any]]: A list of chat dictionaries with metadata.
    """
    conn = _get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, title, model, created_at, updated_at 
        FROM chats 
        ORDER BY updated_at DESC
    """)
    
    chats = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return chats


def get_chat(chat_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve a specific chat session.
    
    Args:
        chat_id: The ID of the chat to retrieve.
        
    Returns:
        Optional[Dict[str, Any]]: The chat dictionary or None if not found.
    """
    conn = _get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT id, title, model, created_at, updated_at FROM chats WHERE id = ?",
        (chat_id,)
    )
    
    row = cursor.fetchone()
    conn.close()
    
    return dict(row) if row else None


def update_chat_title(chat_id: int, title: str) -> None:
    """Update the title of a chat session.
    
    Args:
        chat_id: The ID of the chat to update.
        title: The new title for the chat.
    """
    conn = _get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "UPDATE chats SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (title, chat_id)
    )
    
    conn.commit()
    conn.close()


def delete_chat(chat_id: int) -> None:
    """Delete a chat session and all its messages.
    
    Args:
        chat_id: The ID of the chat to delete.
    """
    conn = _get_connection()
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    
    conn.commit()
    conn.close()


def add_message(chat_id: int, role: str, content: str) -> int:
    """Add a message to a chat session.
    
    Args:
        chat_id: The ID of the chat to add the message to.
        role: The role of the message sender (e.g., 'user', 'assistant', 'system').
        content: The content of the message.
        
    Returns:
        int: The ID of the newly created message.
    """
    conn = _get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)",
        (chat_id, role, content)
    )
    
    message_id = cursor.lastrowid
    
    # Update the chat's updated_at timestamp
    cursor.execute(
        "UPDATE chats SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (chat_id,)
    )
    
    conn.commit()
    conn.close()
    
    return message_id


def get_messages(chat_id: int) -> List[Dict[str, Any]]:
    """Retrieve all messages for a specific chat.
    
    Args:
        chat_id: The ID of the chat.
        
    Returns:
        List[Dict[str, Any]]: A list of message dictionaries.
    """
    conn = _get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute(
        """
        SELECT id, chat_id, role, content, created_at 
        FROM messages 
        WHERE chat_id = ? 
        ORDER BY created_at ASC
        """,
        (chat_id,)
    )
    
    messages = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return messages


def clear_chat_messages(chat_id: int) -> None:
    """Clear all messages from a chat session.
    
    Args:
        chat_id: The ID of the chat to clear.
    """
    conn = _get_connection()
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
    
    # Update the chat's updated_at timestamp
    cursor.execute(
        "UPDATE chats SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (chat_id,)
    )
    
    conn.commit()
    conn.close()


def search_chats(query: str) -> List[Dict[str, Any]]:
    """Search for chats by title or content.
    
    Args:
        query: The search query string.
        
    Returns:
        List[Dict[str, Any]]: A list of matching chat dictionaries.
    """
    conn = _get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute(
        """
        SELECT DISTINCT c.id, c.title, c.model, c.created_at, c.updated_at
        FROM chats c
        LEFT JOIN messages m ON c.id = m.chat_id
        WHERE c.title LIKE ? OR m.content LIKE ?
        ORDER BY c.updated_at DESC
        """,
        (f"%{query}%", f"%{query}%")
    )
    
    chats = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return chats


if __name__ == "__main__":
    # Initialize the database when run directly
    init_database()
    print(f"Database initialized at: {get_db_path()}")
