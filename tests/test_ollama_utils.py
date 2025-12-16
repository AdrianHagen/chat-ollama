"""Tests for the Ollama utilities module.

This module contains comprehensive tests for Ollama model management
and chat streaming functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from chat_ollama.ollama_utils import get_model, chat_model


class TestGetModel:
    """Test suite for the get_model function."""
    
    @patch('chat_ollama.ollama_utils.ollama.show')
    def test_get_model_already_exists(self, mock_show, capsys):
        """Test when model is already available locally.
        
        Args:
            mock_show: Mock for ollama.show.
            capsys: Pytest fixture to capture stdout.
        """
        mock_show.return_value = {"model": "llama2"}
        
        get_model("llama2")
        
        mock_show.assert_called_once_with("llama2")
        captured = capsys.readouterr()
        assert "Model llama2 already pulled." in captured.out
    
    @patch('chat_ollama.ollama_utils.ollama.pull')
    @patch('chat_ollama.ollama_utils.ollama.show')
    def test_get_model_needs_pulling(self, mock_show, mock_pull, capsys):
        """Test when model needs to be pulled.
        
        Args:
            mock_show: Mock for ollama.show.
            mock_pull: Mock for ollama.pull.
            capsys: Pytest fixture to capture stdout.
        """
        mock_show.side_effect = Exception("Model not found")
        mock_pull.return_value = None
        
        get_model("gemma2")
        
        mock_show.assert_called_once_with("gemma2")
        mock_pull.assert_called_once_with("gemma2")
        captured = capsys.readouterr()
        assert "Model gemma2 not found locally." in captured.out
        assert "Pulling gemma2 from ollama." in captured.out
    
    @patch('chat_ollama.ollama_utils.ollama.pull')
    @patch('chat_ollama.ollama_utils.ollama.show')
    def test_get_model_pull_fails(self, mock_show, mock_pull, capsys):
        """Test when model pull fails.
        
        Args:
            mock_show: Mock for ollama.show.
            mock_pull: Mock for ollama.pull.
            capsys: Pytest fixture to capture stdout.
        """
        mock_show.side_effect = Exception("Model not found")
        mock_pull.side_effect = Exception("Network error")
        
        get_model("nonexistent_model")
        
        mock_show.assert_called_once_with("nonexistent_model")
        mock_pull.assert_called_once_with("nonexistent_model")
        captured = capsys.readouterr()
        assert "Model nonexistent_model not found locally." in captured.out
        assert "Model nonexistent_model not found on ollama, please use a different model" in captured.out
    
    @patch('chat_ollama.ollama_utils.ollama.show')
    def test_get_model_various_model_names(self, mock_show):
        """Test get_model with various model name formats.
        
        Args:
            mock_show: Mock for ollama.show.
        """
        mock_show.return_value = {"model": "test"}
        
        model_names = ["llama2", "gemma2:7b", "mistral:latest", "phi-2"]
        
        for model_name in model_names:
            get_model(model_name)
            mock_show.assert_called_with(model_name)
    
    @patch('chat_ollama.ollama_utils.ollama.pull')
    @patch('chat_ollama.ollama_utils.ollama.show')
    def test_get_model_returns_none(self, mock_show, mock_pull):
        """Test that get_model returns None.
        
        Args:
            mock_show: Mock for ollama.show.
            mock_pull: Mock for ollama.pull.
        """
        mock_show.return_value = {"model": "test"}
        
        result = get_model("llama2")
        assert result is None


class TestChatModel:
    """Test suite for the chat_model function."""
    
    @patch('chat_ollama.ollama_utils.ollama.chat')
    @patch('chat_ollama.ollama_utils.get_model')
    def test_chat_model_basic_streaming(self, mock_get_model, mock_chat, capsys):
        """Test basic chat streaming functionality.
        
        Args:
            mock_get_model: Mock for get_model.
            mock_chat: Mock for ollama.chat.
            capsys: Pytest fixture to capture stdout.
        """
        # Create mock message objects
        mock_messages = [
            Mock(message=Mock(content="Hello")),
            Mock(message=Mock(content=" there")),
            Mock(message=Mock(content="!")),
        ]
        mock_chat.return_value = iter(mock_messages)
        
        messages = [{"role": "user", "content": "Hi"}]
        result = list(chat_model("llama2", messages))
        
        mock_get_model.assert_called_once_with(model_name="llama2")
        mock_chat.assert_called_once_with(model="llama2", messages=messages, stream=True)
        assert result == ["Hello", " there", "!"]
        
        captured = capsys.readouterr()
        assert "Hello there!" in captured.out
    
    @patch('chat_ollama.ollama_utils.ollama.chat')
    @patch('chat_ollama.ollama_utils.get_model')
    def test_chat_model_with_conversation_history(self, mock_get_model, mock_chat):
        """Test chat with multiple messages in history.
        
        Args:
            mock_get_model: Mock for get_model.
            mock_chat: Mock for ollama.chat.
        """
        mock_messages = [Mock(message=Mock(content="Response"))]
        mock_chat.return_value = iter(mock_messages)
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        
        result = list(chat_model("gemma2", messages))
        
        mock_chat.assert_called_once_with(model="gemma2", messages=messages, stream=True)
        assert result == ["Response"]
    
    @patch('chat_ollama.ollama_utils.ollama.chat')
    @patch('chat_ollama.ollama_utils.get_model')
    def test_chat_model_empty_response(self, mock_get_model, mock_chat):
        """Test chat with empty response.
        
        Args:
            mock_get_model: Mock for get_model.
            mock_chat: Mock for ollama.chat.
        """
        mock_chat.return_value = iter([])
        
        messages = [{"role": "user", "content": "Hello"}]
        result = list(chat_model("llama2", messages))
        
        assert result == []
    
    @patch('chat_ollama.ollama_utils.ollama.chat')
    @patch('chat_ollama.ollama_utils.get_model')
    def test_chat_model_single_chunk(self, mock_get_model, mock_chat):
        """Test chat with single chunk response.
        
        Args:
            mock_get_model: Mock for get_model.
            mock_chat: Mock for ollama.chat.
        """
        mock_messages = [Mock(message=Mock(content="Complete response"))]
        mock_chat.return_value = iter(mock_messages)
        
        messages = [{"role": "user", "content": "What is 2+2?"}]
        result = list(chat_model("llama2", messages))
        
        assert result == ["Complete response"]
    
    @patch('chat_ollama.ollama_utils.ollama.chat')
    @patch('chat_ollama.ollama_utils.get_model')
    def test_chat_model_many_chunks(self, mock_get_model, mock_chat):
        """Test chat with many small chunks.
        
        Args:
            mock_get_model: Mock for get_model.
            mock_chat: Mock for ollama.chat.
        """
        # Simulate streaming response word by word
        words = ["The", " answer", " is", " 42", "."]
        mock_messages = [Mock(message=Mock(content=word)) for word in words]
        mock_chat.return_value = iter(mock_messages)
        
        messages = [{"role": "user", "content": "Question"}]
        result = list(chat_model("llama2", messages))
        
        assert result == words
        assert "".join(result) == "The answer is 42."
    
    @patch('chat_ollama.ollama_utils.ollama.chat')
    @patch('chat_ollama.ollama_utils.get_model')
    def test_chat_model_different_models(self, mock_get_model, mock_chat):
        """Test chat with different model names.
        
        Args:
            mock_get_model: Mock for get_model.
            mock_chat: Mock for ollama.chat.
        """
        mock_messages = [Mock(message=Mock(content="Response"))]
        mock_chat.return_value = iter(mock_messages)
        
        messages = [{"role": "user", "content": "Test"}]
        models = ["llama2", "gemma2:7b", "mistral", "phi-2"]
        
        for model in models:
            list(chat_model(model, messages))
            mock_get_model.assert_called_with(model_name=model)
    
    @patch('chat_ollama.ollama_utils.ollama.chat')
    @patch('chat_ollama.ollama_utils.get_model')
    def test_chat_model_is_generator(self, mock_get_model, mock_chat):
        """Test that chat_model returns a generator.
        
        Args:
            mock_get_model: Mock for get_model.
            mock_chat: Mock for ollama.chat.
        """
        mock_messages = [Mock(message=Mock(content="Test"))]
        mock_chat.return_value = iter(mock_messages)
        
        messages = [{"role": "user", "content": "Hi"}]
        result = chat_model("llama2", messages)
        
        # Check that it's a generator
        assert hasattr(result, '__iter__')
        assert hasattr(result, '__next__')
    
    @patch('chat_ollama.ollama_utils.ollama.chat')
    @patch('chat_ollama.ollama_utils.get_model')
    def test_chat_model_preserves_message_order(self, mock_get_model, mock_chat):
        """Test that message order is preserved.
        
        Args:
            mock_get_model: Mock for get_model.
            mock_chat: Mock for ollama.chat.
        """
        mock_messages = [Mock(message=Mock(content="Response"))]
        mock_chat.return_value = iter(mock_messages)
        
        messages = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Second"},
            {"role": "user", "content": "Third"},
        ]
        
        list(chat_model("llama2", messages))
        
        # Verify the exact messages list was passed
        call_args = mock_chat.call_args
        assert call_args[1]["messages"] == messages
    
    @patch('chat_ollama.ollama_utils.ollama.chat')
    @patch('chat_ollama.ollama_utils.get_model')
    def test_chat_model_stream_flag(self, mock_get_model, mock_chat):
        """Test that stream flag is always set to True.
        
        Args:
            mock_get_model: Mock for get_model.
            mock_chat: Mock for ollama.chat.
        """
        mock_messages = [Mock(message=Mock(content="Response"))]
        mock_chat.return_value = iter(mock_messages)
        
        messages = [{"role": "user", "content": "Test"}]
        list(chat_model("llama2", messages))
        
        call_args = mock_chat.call_args
        assert call_args[1]["stream"] is True
    
    @patch('chat_ollama.ollama_utils.ollama.chat')
    @patch('chat_ollama.ollama_utils.get_model')
    def test_chat_model_with_unicode_content(self, mock_get_model, mock_chat):
        """Test chat with Unicode characters in messages.
        
        Args:
            mock_get_model: Mock for get_model.
            mock_chat: Mock for ollama.chat.
        """
        mock_messages = [Mock(message=Mock(content="你好 🌍"))]
        mock_chat.return_value = iter(mock_messages)
        
        messages = [{"role": "user", "content": "Hello 世界"}]
        result = list(chat_model("llama2", messages))
        
        assert result == ["你好 🌍"]
    
    @patch('chat_ollama.ollama_utils.ollama.chat')
    @patch('chat_ollama.ollama_utils.get_model')
    def test_chat_model_exception_propagation(self, mock_get_model, mock_chat):
        """Test that exceptions from ollama.chat are propagated.
        
        Args:
            mock_get_model: Mock for get_model.
            mock_chat: Mock for ollama.chat.
        """
        mock_chat.side_effect = Exception("Connection error")
        
        messages = [{"role": "user", "content": "Test"}]
        
        with pytest.raises(Exception, match="Connection error"):
            list(chat_model("llama2", messages))


class TestIntegration:
    """Integration tests for ollama_utils functions."""
    
    @patch('chat_ollama.ollama_utils.ollama.chat')
    @patch('chat_ollama.ollama_utils.ollama.show')
    def test_full_chat_flow_model_exists(self, mock_show, mock_chat, capsys):
        """Test complete flow when model exists.
        
        Args:
            mock_show: Mock for ollama.show.
            mock_chat: Mock for ollama.chat.
            capsys: Pytest fixture to capture stdout.
        """
        mock_show.return_value = {"model": "llama2"}
        mock_messages = [
            Mock(message=Mock(content="Sure")),
            Mock(message=Mock(content=", I can help!")),
        ]
        mock_chat.return_value = iter(mock_messages)
        
        messages = [{"role": "user", "content": "Can you help?"}]
        result = list(chat_model("llama2", messages))
        
        assert result == ["Sure", ", I can help!"]
        captured = capsys.readouterr()
        assert "Model llama2 already pulled." in captured.out
        assert "Sure, I can help!" in captured.out
    
    @patch('chat_ollama.ollama_utils.ollama.chat')
    @patch('chat_ollama.ollama_utils.ollama.pull')
    @patch('chat_ollama.ollama_utils.ollama.show')
    def test_full_chat_flow_model_needs_pull(self, mock_show, mock_pull, mock_chat, capsys):
        """Test complete flow when model needs to be pulled.
        
        Args:
            mock_show: Mock for ollama.show.
            mock_pull: Mock for ollama.pull.
            mock_chat: Mock for ollama.chat.
            capsys: Pytest fixture to capture stdout.
        """
        mock_show.side_effect = Exception("Not found")
        mock_pull.return_value = None
        mock_messages = [Mock(message=Mock(content="Hello!"))]
        mock_chat.return_value = iter(mock_messages)
        
        messages = [{"role": "user", "content": "Hi"}]
        result = list(chat_model("gemma2", messages))
        
        assert result == ["Hello!"]
        captured = capsys.readouterr()
        assert "Model gemma2 not found locally." in captured.out
        assert "Pulling gemma2 from ollama." in captured.out
        assert "Hello!" in captured.out
