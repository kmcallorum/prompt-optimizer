"""Extended tests for storage module."""

from pathlib import Path

from prompt_optimizer.storage import PromptStorage


class TestSaveResults:
    """Tests for save_results function."""

    def test_save_results(self, tmp_path: Path) -> None:
        """Test saving optimization results."""
        storage = PromptStorage(tmp_path / ".prompt-optimizer")

        results_dict = {
            "base_prompt_name": "test",
            "variants_tested": 2,
            "total_cost": 0.005,
            "best_variant": {"strategy": "concise"},
        }

        result_path = storage.save_results(results_dict, "optimization-run")

        assert result_path.exists()
        assert "results_optimization-run" in result_path.name
        assert result_path.suffix == ".json"

        content = result_path.read_text()
        assert "test" in content
        assert "concise" in content

    def test_save_results_with_special_types(self, tmp_path: Path) -> None:
        """Test saving results with special types that need serialization."""
        storage = PromptStorage(tmp_path / ".prompt-optimizer")

        from datetime import datetime

        results_dict = {
            "timestamp": datetime.now(),
            "data": {"nested": "value"},
        }

        result_path = storage.save_results(results_dict, "special-types")
        assert result_path.exists()
