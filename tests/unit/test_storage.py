"""Unit tests for storage module."""

from pathlib import Path

import pytest

from prompt_optimizer.prompt import Prompt
from prompt_optimizer.storage import PromptStorage, generate_version_id

pytestmark = pytest.mark.unit


class TestGenerateVersionId:
    """Tests for generate_version_id function."""

    def test_generates_string(self) -> None:
        """Test that it generates a string."""
        version_id = generate_version_id()
        assert isinstance(version_id, str)
        assert len(version_id) > 0

    def test_unique_ids(self) -> None:
        """Test that IDs are unique."""
        ids = [generate_version_id() for _ in range(10)]
        assert len(set(ids)) == 10


class TestPromptStorage:
    """Tests for PromptStorage class."""

    def test_storage_creation(self, tmp_path: Path) -> None:
        """Test storage directory creation."""
        storage_path = tmp_path / ".prompt-optimizer"
        PromptStorage(storage_path)
        assert storage_path.exists()

    def test_save_prompt(self, tmp_path: Path) -> None:
        """Test saving a prompt."""
        storage = PromptStorage(tmp_path / ".prompt-optimizer")
        prompt = Prompt(
            template="Test {{ var }}",
            name="test-prompt",
        )
        version_id = storage.save_prompt(prompt)
        assert version_id is not None

        saved_files = list((tmp_path / ".prompt-optimizer").glob("*.json"))
        assert len(saved_files) == 1

    def test_load_prompt(self, tmp_path: Path) -> None:
        """Test loading a prompt."""
        storage = PromptStorage(tmp_path / ".prompt-optimizer")
        prompt = Prompt(
            template="Test {{ var }}",
            name="load-test",
        )
        version_id = storage.save_prompt(prompt)

        loaded = storage.load_prompt("load-test", version_id)
        assert loaded.template == prompt.template
        assert loaded.name == prompt.name

    def test_load_latest(self, tmp_path: Path) -> None:
        """Test loading latest version."""
        import time

        storage = PromptStorage(tmp_path / ".prompt-optimizer")

        # Save two versions with delay to ensure different timestamps
        prompt1 = Prompt(template="Version 1", name="latest-test")
        storage.save_prompt(prompt1)

        time.sleep(0.1)  # Ensure different mtime

        prompt2 = Prompt(template="Version 2", name="latest-test")
        storage.save_prompt(prompt2)

        # Load latest (should be version 2)
        loaded = storage.load_prompt("latest-test")
        assert loaded.template == "Version 2"

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        """Test loading nonexistent prompt raises error."""
        storage = PromptStorage(tmp_path / ".prompt-optimizer")
        with pytest.raises(FileNotFoundError):
            storage.load_prompt("nonexistent")

    def test_history(self, tmp_path: Path) -> None:
        """Test getting version history."""
        storage = PromptStorage(tmp_path / ".prompt-optimizer")

        # Save multiple versions
        for i in range(3):
            prompt = Prompt(template=f"Version {i}", name="history-test")
            storage.save_prompt(prompt)

        history = storage.history("history-test")
        assert len(history) == 3

    def test_list_prompts(self, tmp_path: Path) -> None:
        """Test listing all prompts."""
        storage = PromptStorage(tmp_path / ".prompt-optimizer")

        for name in ["prompt-a", "prompt-b", "prompt-c"]:
            prompt = Prompt(template=f"Template for {name}", name=name)
            storage.save_prompt(prompt)

        prompts = storage.list_prompts()
        assert len(prompts) == 3
        assert "prompt-a" in prompts
        assert "prompt-b" in prompts
        assert "prompt-c" in prompts
