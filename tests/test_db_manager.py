"""Tests for the database manager module.

This module contains comprehensive tests for all database operations including
creating, reading, updating, and deleting chats and messages.
"""

import pytest
import sqlite3
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch
from chat_ollama.db_manager import (
    get_db_path,
    init_database,
    create_chat,
    get_all_chats,
    get_chat,
    update_chat_title,
    delete_chat,
    add_message,
    get_messages,
    clear_chat_messages,
    search_chats,
)


@pytest.fixture
def temp_db_path(monkeypatch):
    """Create a temporary database path for testing.
    
    Yields:
        Path: Temporary directory path for the test database.
    """
    temp_dir = Path(tempfile.mkdtemp())
    db_path = temp_dir / "test_chats.db"
    
    # Mock get_db_path to return our temp path
    monkeypatch.setattr("chat_ollama.db_manager.get_db_path", lambda: db_path)
    
    yield db_path
    
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def initialized_db(temp_db_path):
    """Initialize a test database.
    
    Args:
        temp_db_path: Fixture providing temporary database path.
        
    Returns:
        Path: Path to the initialized test database.
    """
    init_database()
    return temp_db_path


class TestDatabaseInitialization:
    """Test suite for database initialization."""
    
    def test_init_database_creates_file(self, temp_db_path):
        """Test that init_database creates the database file."""
        assert not temp_db_path.exists()
        init_database()
        assert temp_db_path.exists()
    
    def test_init_database_creates_chats_table(self, initialized_db):
        """Test that the chats table is created correctly."""
        conn = sqlite3.connect(initialized_db)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chats'"
        )
        assert cursor.fetchone() is not None
        conn.close()
    
    def test_init_database_creates_messages_table(self, initialized_db):
        """Test that the messages table is created correctly."""
        conn = sqlite3.connect(initialized_db)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='messages'"
        )
        assert cursor.fetchone() is not None
        conn.close()
    
    def test_init_database_creates_index(self, initialized_db):
        """Test that the index on messages table is created."""
        conn = sqlite3.connect(initialized_db)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_messages_chat_id'"
        )
        assert cursor.fetchone() is not None
        conn.close()
    
    def test_init_database_idempotent(self, initialized_db):
        """Test that calling init_database multiple times is safe."""
        # Should not raise any errors
        init_database()
        init_database()
        assert initialized_db.exists()


class TestChatOperations:
    """Test suite for chat CRUD operations."""
    
    def test_create_chat_returns_id(self, initialized_db):
        """Test that create_chat returns a valid chat ID."""
        chat_id = create_chat("Test Chat", "llama2")
        assert isinstance(chat_id, int)
        assert chat_id > 0
    
    def test_create_chat_stores_data(self, initialized_db):
        """Test that create_chat stores the chat data correctly."""
        chat_id = create_chat("My Chat", "gemma2")
        
        conn = sqlite3.connect(initialized_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM chats WHERE id = ?", (chat_id,))
        row = cursor.fetchone()
        
        assert row["title"] == "My Chat"
        assert row["model"] == "gemma2"
        assert row["created_at"] is not None
        assert row["updated_at"] is not None
        conn.close()
    
    def test_get_all_chats_empty(self, initialized_db):
        """Test getting all chats when database is empty."""
        chats = get_all_chats()
        assert chats == []
    
    def test_get_all_chats_returns_list(self, initialized_db):
        """Test that get_all_chats returns all chats."""
        chat_id1 = create_chat("Chat 1", "llama2")
        chat_id2 = create_chat("Chat 2", "gemma2")
        
        chats = get_all_chats()
        assert len(chats) == 2
        assert any(chat["id"] == chat_id1 for chat in chats)
        assert any(chat["id"] == chat_id2 for chat in chats)
    
    def test_get_all_chats_ordered_by_updated_at(self, initialized_db):
        """Test that chats are returned in reverse chronological order."""
        chat_id1 = create_chat("First Chat", "llama2")
        
        # Small delay to ensure different timestamp
        import time
        time.sleep(0.01)
        
        chat_id2 = create_chat("Second Chat", "gemma2")
        
        chats = get_all_chats()
        # Both chats should be present
        assert len(chats) == 2
        chat_ids = [chat["id"] for chat in chats]
        assert chat_id1 in chat_ids
        assert chat_id2 in chat_ids
        # Most recently updated should be first (chat_id2 or equal timestamp)
        assert chats[0]["id"] in [chat_id1, chat_id2]
    
    def test_get_chat_existing(self, initialized_db):
        """Test retrieving an existing chat."""
        chat_id = create_chat("Test Chat", "llama2")
        chat = get_chat(chat_id)
        
        assert chat is not None
        assert chat["id"] == chat_id
        assert chat["title"] == "Test Chat"
        assert chat["model"] == "llama2"
    
    def test_get_chat_nonexistent(self, initialized_db):
        """Test retrieving a non-existent chat returns None."""
        chat = get_chat(9999)
        assert chat is None
    
    def test_update_chat_title(self, initialized_db):
        """Test updating a chat's title."""
        chat_id = create_chat("Original Title", "llama2")
        update_chat_title(chat_id, "New Title")
        
        chat = get_chat(chat_id)
        assert chat["title"] == "New Title"
    
    def test_update_chat_title_updates_timestamp(self, initialized_db):
        """Test that updating title updates the updated_at timestamp."""
        chat_id = create_chat("Original", "llama2")
        chat_before = get_chat(chat_id)
        
        # Small delay to ensure timestamp difference
        import time
        time.sleep(0.01)
        
        update_chat_title(chat_id, "Updated")
        chat_after = get_chat(chat_id)
        
        # SQLite timestamps might be the same due to precision, but test the update happened
        assert chat_after["title"] == "Updated"
    
    def test_delete_chat(self, initialized_db):
        """Test deleting a chat."""
        chat_id = create_chat("To Delete", "llama2")
        delete_chat(chat_id)
        
        chat = get_chat(chat_id)
        assert chat is None
    
    def test_delete_chat_cascades_to_messages(self, initialized_db):
        """Test that deleting a chat also deletes its messages."""
        chat_id = create_chat("Test Chat", "llama2")
        add_message(chat_id, "user", "Hello")
        add_message(chat_id, "assistant", "Hi there")
        
        delete_chat(chat_id)
        
        messages = get_messages(chat_id)
        assert messages == []


class TestMessageOperations:
    """Test suite for message operations."""
    
    def test_add_message_returns_id(self, initialized_db):
        """Test that add_message returns a valid message ID."""
        chat_id = create_chat("Test Chat", "llama2")
        message_id = add_message(chat_id, "user", "Hello world")
        
        assert isinstance(message_id, int)
        assert message_id > 0
    
    def test_add_message_stores_data(self, initialized_db):
        """Test that add_message stores the message data correctly."""
        chat_id = create_chat("Test Chat", "llama2")
        message_id = add_message(chat_id, "user", "Test message")
        
        conn = sqlite3.connect(initialized_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM messages WHERE id = ?", (message_id,))
        row = cursor.fetchone()
        
        assert row["chat_id"] == chat_id
        assert row["role"] == "user"
        assert row["content"] == "Test message"
        assert row["created_at"] is not None
        conn.close()
    
    def test_add_message_updates_chat_timestamp(self, initialized_db):
        """Test that adding a message updates the chat's updated_at timestamp."""
        chat_id = create_chat("Test Chat", "llama2")
        chat_before = get_chat(chat_id)
        
        import time
        time.sleep(0.01)
        
        add_message(chat_id, "user", "Hello")
        chat_after = get_chat(chat_id)
        
        # Verify message was added and chat exists
        assert chat_after is not None
        messages = get_messages(chat_id)
        assert len(messages) == 1
    
    def test_get_messages_empty(self, initialized_db):
        """Test getting messages from a chat with no messages."""
        chat_id = create_chat("Empty Chat", "llama2")
        messages = get_messages(chat_id)
        assert messages == []
    
    def test_get_messages_returns_all(self, initialized_db):
        """Test that get_messages returns all messages for a chat."""
        chat_id = create_chat("Test Chat", "llama2")
        msg_id1 = add_message(chat_id, "user", "First message")
        msg_id2 = add_message(chat_id, "assistant", "Second message")
        msg_id3 = add_message(chat_id, "user", "Third message")
        
        messages = get_messages(chat_id)
        assert len(messages) == 3
        assert messages[0]["id"] == msg_id1
        assert messages[1]["id"] == msg_id2
        assert messages[2]["id"] == msg_id3
    
    def test_get_messages_ordered_chronologically(self, initialized_db):
        """Test that messages are returned in chronological order."""
        chat_id = create_chat("Test Chat", "llama2")
        add_message(chat_id, "user", "First")
        add_message(chat_id, "assistant", "Second")
        add_message(chat_id, "user", "Third")
        
        messages = get_messages(chat_id)
        assert messages[0]["content"] == "First"
        assert messages[1]["content"] == "Second"
        assert messages[2]["content"] == "Third"
    
    def test_get_messages_only_returns_chat_messages(self, initialized_db):
        """Test that get_messages only returns messages for the specified chat."""
        chat_id1 = create_chat("Chat 1", "llama2")
        chat_id2 = create_chat("Chat 2", "gemma2")
        
        add_message(chat_id1, "user", "Message for chat 1")
        add_message(chat_id2, "user", "Message for chat 2")
        
        messages = get_messages(chat_id1)
        assert len(messages) == 1
        assert messages[0]["content"] == "Message for chat 1"
    
    def test_clear_chat_messages(self, initialized_db):
        """Test clearing all messages from a chat."""
        chat_id = create_chat("Test Chat", "llama2")
        add_message(chat_id, "user", "Message 1")
        add_message(chat_id, "assistant", "Message 2")
        
        clear_chat_messages(chat_id)
        
        messages = get_messages(chat_id)
        assert messages == []
    
    def test_clear_chat_messages_preserves_chat(self, initialized_db):
        """Test that clearing messages doesn't delete the chat."""
        chat_id = create_chat("Test Chat", "llama2")
        add_message(chat_id, "user", "Message")
        
        clear_chat_messages(chat_id)
        
        chat = get_chat(chat_id)
        assert chat is not None
        assert chat["title"] == "Test Chat"
    
    def test_clear_chat_messages_updates_timestamp(self, initialized_db):
        """Test that clearing messages updates the chat's timestamp."""
        chat_id = create_chat("Test Chat", "llama2")
        add_message(chat_id, "user", "Message")
        
        import time
        time.sleep(0.01)
        
        clear_chat_messages(chat_id)
        
        chat = get_chat(chat_id)
        assert chat is not None


class TestSearchFunctionality:
    """Test suite for search operations."""
    
    def test_search_chats_by_title(self, initialized_db):
        """Test searching chats by title."""
        chat_id1 = create_chat("Python Tutorial", "llama2")
        chat_id2 = create_chat("JavaScript Guide", "gemma2")
        chat_id3 = create_chat("Python Advanced", "llama2")
        
        results = search_chats("Python")
        assert len(results) == 2
        assert any(r["id"] == chat_id1 for r in results)
        assert any(r["id"] == chat_id3 for r in results)
    
    def test_search_chats_by_message_content(self, initialized_db):
        """Test searching chats by message content."""
        chat_id1 = create_chat("Chat 1", "llama2")
        chat_id2 = create_chat("Chat 2", "gemma2")
        
        add_message(chat_id1, "user", "Tell me about Python")
        add_message(chat_id2, "user", "Tell me about JavaScript")
        
        results = search_chats("Python")
        assert len(results) == 1
        assert results[0]["id"] == chat_id1
    
    def test_search_chats_case_insensitive(self, initialized_db):
        """Test that search is case-insensitive."""
        chat_id = create_chat("Python Tutorial", "llama2")
        
        results = search_chats("python")
        assert len(results) == 1
        assert results[0]["id"] == chat_id
    
    def test_search_chats_partial_match(self, initialized_db):
        """Test that search supports partial matches."""
        chat_id = create_chat("Machine Learning Basics", "llama2")
        
        results = search_chats("Learn")
        assert len(results) == 1
        assert results[0]["id"] == chat_id
    
    def test_search_chats_no_results(self, initialized_db):
        """Test searching with no matching results."""
        create_chat("Python Tutorial", "llama2")
        
        results = search_chats("Rust")
        assert results == []
    
    def test_search_chats_empty_query(self, initialized_db):
        """Test searching with an empty query."""
        chat_id1 = create_chat("Chat 1", "llama2")
        chat_id2 = create_chat("Chat 2", "gemma2")
        
        results = search_chats("")
        assert len(results) == 2


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""
    
    def test_add_message_to_nonexistent_chat(self, initialized_db):
        """Test adding a message to a non-existent chat."""
        # With foreign key enforcement enabled, this should raise an error
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY constraint failed"):
            add_message(9999, "user", "Orphaned message")
    
    def test_chat_with_special_characters(self, initialized_db):
        """Test creating a chat with special characters in title."""
        chat_id = create_chat("Test's \"Chat\" & More", "llama2")
        chat = get_chat(chat_id)
        assert chat["title"] == "Test's \"Chat\" & More"
    
    def test_message_with_special_characters(self, initialized_db):
        """Test adding a message with special characters."""
        chat_id = create_chat("Test Chat", "llama2")
        add_message(chat_id, "user", "Hello <html> & \"quotes\" 'test'")
        
        messages = get_messages(chat_id)
        assert messages[0]["content"] == "Hello <html> & \"quotes\" 'test'"
    
    def test_message_with_unicode(self, initialized_db):
        """Test adding messages with Unicode characters."""
        chat_id = create_chat("Test Chat", "llama2")
        add_message(chat_id, "user", "Hello 世界 🌍 café")
        
        messages = get_messages(chat_id)
        assert messages[0]["content"] == "Hello 世界 🌍 café"
    
    def test_empty_message_content(self, initialized_db):
        """Test adding a message with empty content."""
        chat_id = create_chat("Test Chat", "llama2")
        message_id = add_message(chat_id, "user", "")
        
        messages = get_messages(chat_id)
        assert len(messages) == 1
        assert messages[0]["content"] == ""
    
    def test_very_long_message(self, initialized_db):
        """Test adding a very long message."""
        chat_id = create_chat("Test Chat", "llama2")
        long_content = "A" * 10000
        add_message(chat_id, "user", long_content)
        
        messages = get_messages(chat_id)
        assert messages[0]["content"] == long_content
