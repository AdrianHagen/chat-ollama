"""Tests for frontend utility functions.

This module contains tests for the frontend utility functions including
chat title generation.
"""

import pytest
from unittest.mock import Mock, patch
from chat_ollama.frontend.frontend_utils import generate_chat_title


class TestGenerateChatTitle:
    """Test suite for the generate_chat_title function."""
    
    @patch('chat_ollama.frontend.frontend_utils.ollama.chat')
    def test_generate_chat_title_basic(self, mock_chat):
        """Test basic chat title generation.
        
        Args:
            mock_chat: Mock for ollama.chat.
        """
        mock_chat.return_value = {
            'message': {'content': 'Python List Sorting'}
        }
        
        title = generate_chat_title(
            "How do I sort a list in Python?",
            "gemma3:12b"
        )
        
        assert title == "Python List Sorting"
        mock_chat.assert_called_once()
        call_args = mock_chat.call_args
        assert call_args[1]['model'] == "gemma3:12b"
        assert call_args[1]['stream'] is False
    
    @patch('chat_ollama.frontend.frontend_utils.ollama.chat')
    def test_generate_chat_title_with_punctuation(self, mock_chat):
        """Test title generation removes punctuation.
        
        Args:
            mock_chat: Mock for ollama.chat.
        """
        mock_chat.return_value = {
            'message': {'content': 'Machine Learning Basics.'}
        }
        
        title = generate_chat_title(
            "What is machine learning?",
            "llama2"
        )
        
        assert title == "Machine Learning Basics"
    
    @patch('chat_ollama.frontend.frontend_utils.ollama.chat')
    def test_generate_chat_title_more_than_three_words(self, mock_chat):
        """Test that only first 3 words are returned.
        
        Args:
            mock_chat: Mock for ollama.chat.
        """
        mock_chat.return_value = {
            'message': {'content': 'Python Programming Language Tutorial Guide'}
        }
        
        title = generate_chat_title(
            "Tell me about Python programming",
            "gemma3:12b"
        )
        
        assert title == "Python Programming Language"
    
    @patch('chat_ollama.frontend.frontend_utils.ollama.chat')
    def test_generate_chat_title_less_than_three_words(self, mock_chat):
        """Test title generation with less than 3 words.
        
        Args:
            mock_chat: Mock for ollama.chat.
        """
        mock_chat.return_value = {
            'message': {'content': 'Python Programming'}
        }
        
        title = generate_chat_title(
            "What is Python?",
            "llama2"
        )
        
        assert title == "Python Programming"
    
    @patch('chat_ollama.frontend.frontend_utils.ollama.chat')
    def test_generate_chat_title_with_extra_whitespace(self, mock_chat):
        """Test title generation handles extra whitespace.
        
        Args:
            mock_chat: Mock for ollama.chat.
        """
        mock_chat.return_value = {
            'message': {'content': '  Machine   Learning   Basics  '}
        }
        
        title = generate_chat_title(
            "Tell me about ML",
            "gemma3:12b"
        )
        
        # Should handle multiple spaces between words
        words = title.split()
        assert len(words) <= 3
    
    @patch('chat_ollama.frontend.frontend_utils.ollama.chat')
    def test_generate_chat_title_empty_message(self, mock_chat):
        """Test title generation with empty message.
        
        Args:
            mock_chat: Mock for ollama.chat.
        """
        title = generate_chat_title("", "gemma3:12b")
        
        assert title == "New Chat"
        mock_chat.assert_not_called()
    
    @patch('chat_ollama.frontend.frontend_utils.ollama.chat')
    def test_generate_chat_title_whitespace_only(self, mock_chat):
        """Test title generation with whitespace-only message.
        
        Args:
            mock_chat: Mock for ollama.chat.
        """
        title = generate_chat_title("   ", "gemma3:12b")
        
        assert title == "New Chat"
        mock_chat.assert_not_called()
    
    @patch('chat_ollama.frontend.frontend_utils.ollama.chat')
    def test_generate_chat_title_api_error(self, mock_chat, capsys):
        """Test title generation when API call fails.
        
        Args:
            mock_chat: Mock for ollama.chat.
            capsys: Pytest fixture to capture stdout.
        """
        mock_chat.side_effect = Exception("API Error")
        
        title = generate_chat_title(
            "How do I sort a list?",
            "gemma3:12b"
        )
        
        # Should fallback to first 3 words of message
        assert title == "How do I"
        
        captured = capsys.readouterr()
        assert "Error generating chat title" in captured.out
    
    @patch('chat_ollama.frontend.frontend_utils.ollama.chat')
    def test_generate_chat_title_empty_response(self, mock_chat):
        """Test title generation when API returns empty content.
        
        Args:
            mock_chat: Mock for ollama.chat.
        """
        mock_chat.return_value = {
            'message': {'content': ''}
        }
        
        title = generate_chat_title(
            "Test message",
            "gemma3:12b"
        )
        
        assert title == "New Chat"
    
    @patch('chat_ollama.frontend.frontend_utils.ollama.chat')
    def test_generate_chat_title_different_models(self, mock_chat):
        """Test title generation with different models.
        
        Args:
            mock_chat: Mock for ollama.chat.
        """
        mock_chat.return_value = {
            'message': {'content': 'Test Title Here'}
        }
        
        models = ["llama2", "gemma3:12b", "mistral", "phi-2"]
        
        for model in models:
            title = generate_chat_title("Test message", model)
            assert title == "Test Title Here"
            # Check that the correct model was used
            call_args = mock_chat.call_args
            assert call_args[1]['model'] == model
    
    @patch('chat_ollama.frontend.frontend_utils.ollama.chat')
    def test_generate_chat_title_long_message(self, mock_chat):
        """Test title generation with a long message.
        
        Args:
            mock_chat: Mock for ollama.chat.
        """
        mock_chat.return_value = {
            'message': {'content': 'Database Query Optimization'}
        }
        
        long_message = (
            "I need help understanding how to optimize database queries. "
            "Specifically, I'm working with PostgreSQL and have some slow queries. "
            "Can you explain indexing strategies?"
        )
        
        title = generate_chat_title(long_message, "gemma3:12b")
        
        assert title == "Database Query Optimization"
        # Verify the full message was sent to the LLM
        call_args = mock_chat.call_args
        messages = call_args[1]['messages']
        assert long_message in messages[1]['content']
    
    @patch('chat_ollama.frontend.frontend_utils.ollama.chat')
    def test_generate_chat_title_with_special_characters(self, mock_chat):
        """Test title generation with special characters in message.
        
        Args:
            mock_chat: Mock for ollama.chat.
        """
        mock_chat.return_value = {
            'message': {'content': 'Python Code Help'}
        }
        
        title = generate_chat_title(
            "How do I use @decorators in Python?",
            "gemma3:12b"
        )
        
        assert title == "Python Code Help"
    
    @patch('chat_ollama.frontend.frontend_utils.ollama.chat')
    def test_generate_chat_title_system_prompt(self, mock_chat):
        """Test that system prompt is correctly formatted.
        
        Args:
            mock_chat: Mock for ollama.chat.
        """
        mock_chat.return_value = {
            'message': {'content': 'Test Title Here'}
        }
        
        generate_chat_title("Test message", "gemma3:12b")
        
        call_args = mock_chat.call_args
        messages = call_args[1]['messages']
        
        # Check system message
        assert messages[0]['role'] == 'system'
        assert '3 words' in messages[0]['content'].lower()
        
        # Check user message
        assert messages[1]['role'] == 'user'
        assert 'Test message' in messages[1]['content']
    
    @patch('chat_ollama.frontend.frontend_utils.ollama.chat')
    def test_generate_chat_title_fallback_short_message(self, mock_chat):
        """Test fallback with message shorter than 3 words.
        
        Args:
            mock_chat: Mock for ollama.chat.
        """
        mock_chat.side_effect = Exception("Error")
        
        title = generate_chat_title("Hello", "gemma3:12b")
        
        assert title == "Hello"
    
    @patch('chat_ollama.frontend.frontend_utils.ollama.chat')
    def test_generate_chat_title_unicode_content(self, mock_chat):
        """Test title generation with Unicode characters.
        
        Args:
            mock_chat: Mock for ollama.chat.
        """
        mock_chat.return_value = {
            'message': {'content': 'Python 编程 Tutorial'}
        }
        
        title = generate_chat_title(
            "How to learn Python 编程?",
            "gemma3:12b"
        )
        
        assert title == "Python 编程 Tutorial"
