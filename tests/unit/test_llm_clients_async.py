"""Tests for LLM client async generate methods."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prompt_optimizer.llm_clients.anthropic import AnthropicClient
from prompt_optimizer.llm_clients.ollama import OllamaClient
from prompt_optimizer.llm_clients.openai import OpenAIClient

pytestmark = pytest.mark.unit


class TestAnthropicClientAsync:
    """Async tests for AnthropicClient."""

    @pytest.mark.asyncio
    async def test_generate(self) -> None:
        """Test generate method."""
        client = AnthropicClient(api_key="test-key")

        # Mock the response
        mock_content = MagicMock()
        mock_content.text = "Test response"

        mock_response = MagicMock()
        mock_response.content = [mock_content]

        with patch.object(
            client.client.messages,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await client.generate(
                prompt="Test prompt",
                system="Be helpful",
                max_tokens=100,
                temperature=0.5,
            )

        assert result == "Test response"

    @pytest.mark.asyncio
    async def test_generate_without_system(self) -> None:
        """Test generate without system message."""
        client = AnthropicClient(api_key="test-key")

        mock_content = MagicMock()
        mock_content.text = "Response"

        mock_response = MagicMock()
        mock_response.content = [mock_content]

        with patch.object(
            client.client.messages,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await client.generate(prompt="Test")

        assert result == "Response"

    @pytest.mark.asyncio
    async def test_generate_content_without_text_attr(self) -> None:
        """Test generate when content has no text attribute."""
        client = AnthropicClient(api_key="test-key")

        # Mock content without text attribute
        mock_content = "plain string content"

        mock_response = MagicMock()
        mock_response.content = [mock_content]

        with patch.object(
            client.client.messages,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await client.generate(prompt="Test")

        assert result == "plain string content"


class TestOpenAIClientAsync:
    """Async tests for OpenAIClient."""

    @pytest.mark.asyncio
    async def test_generate(self) -> None:
        """Test generate method."""
        client = OpenAIClient(api_key="test-key")

        mock_message = MagicMock()
        mock_message.content = "OpenAI response"

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch.object(
            client.client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await client.generate(
                prompt="Test prompt",
                system="Be helpful",
            )

        assert result == "OpenAI response"

    @pytest.mark.asyncio
    async def test_generate_without_system(self) -> None:
        """Test generate without system message."""
        client = OpenAIClient(api_key="test-key")

        mock_message = MagicMock()
        mock_message.content = "Response"

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch.object(
            client.client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await client.generate(prompt="Test")

        assert result == "Response"

    @pytest.mark.asyncio
    async def test_generate_empty_content(self) -> None:
        """Test generate with None content."""
        client = OpenAIClient(api_key="test-key")

        mock_message = MagicMock()
        mock_message.content = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch.object(
            client.client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await client.generate(prompt="Test")

        assert result == ""


class TestOllamaClientAsync:
    """Async tests for OllamaClient."""

    @pytest.mark.asyncio
    async def test_generate(self) -> None:
        """Test generate method."""
        client = OllamaClient()

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Ollama response"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await client.generate(
                prompt="Test prompt",
                system="Be helpful",
                max_tokens=100,
                temperature=0.5,
            )

        assert result == "Ollama response"
