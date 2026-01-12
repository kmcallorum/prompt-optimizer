"""Prompt version storage."""

import json
import uuid
from datetime import datetime
from pathlib import Path

from prompt_optimizer.prompt import Prompt


def generate_version_id() -> str:
    """Generate a unique version ID."""
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]


class PromptStorage:
    """Version control for prompts."""

    def __init__(self, storage_path: Path | None = None) -> None:
        self.storage_path = storage_path or Path(".prompt-optimizer")
        self.storage_path.mkdir(exist_ok=True)

    def save_prompt(self, prompt: Prompt, version: str | None = None) -> str:
        """Save prompt and return version ID."""
        version_id = version or generate_version_id()
        prompt_file = self.storage_path / f"{prompt.name}_{version_id}.json"
        prompt_file.write_text(prompt.model_dump_json(indent=2))
        return version_id

    def load_prompt(self, name: str, version: str | None = None) -> Prompt:
        """Load specific version or latest."""
        if version:
            prompt_file = self.storage_path / f"{name}_{version}.json"
        else:
            versions = list(self.storage_path.glob(f"{name}_*.json"))
            if not versions:
                raise FileNotFoundError(f"No prompts found with name: {name}")
            prompt_file = max(versions, key=lambda p: p.stat().st_mtime)

        return Prompt.model_validate_json(prompt_file.read_text())

    def history(self, name: str) -> list[dict[str, str | float | int]]:
        """Get version history."""
        versions = list(self.storage_path.glob(f"{name}_*.json"))
        return [
            {
                "version": v.stem.replace(f"{name}_", ""),
                "modified": v.stat().st_mtime,
                "size": v.stat().st_size,
            }
            for v in sorted(versions, key=lambda p: p.stat().st_mtime, reverse=True)
        ]

    def list_prompts(self) -> list[str]:
        """List all unique prompt names."""
        files = list(self.storage_path.glob("*.json"))
        names = set()
        for f in files:
            # Version ID format: YYYYMMDD_HHMMSS_uuid (3 parts with _)
            # Filename format: promptname_YYYYMMDD_HHMMSS_uuid.json
            parts = f.stem.rsplit("_", 3)
            if len(parts) >= 4:
                names.add(parts[0])
        return sorted(names)

    def save_results(self, results_dict: dict[str, object], name: str) -> Path:
        """Save optimization results."""
        version_id = generate_version_id()
        results_file = self.storage_path / f"results_{name}_{version_id}.json"
        results_file.write_text(json.dumps(results_dict, indent=2, default=str))
        return results_file
