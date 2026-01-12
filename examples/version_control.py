#!/usr/bin/env python3
"""Example of prompt version control."""

from pathlib import Path
from datetime import datetime

from prompt_optimizer import Prompt
from prompt_optimizer.storage import PromptStorage


def main() -> None:
    """Demonstrate prompt version control."""
    # Initialize storage
    storage = PromptStorage(Path(".prompt-optimizer"))

    print("Prompt Version Control Example")
    print("=" * 40)

    # Create initial prompt
    prompt_v1 = Prompt(
        template="Answer the question: {{ question }}",
        system_message="You are a helpful assistant.",
        name="qa-prompt",
        metadata={"author": "developer", "version": "1.0"},
    )

    # Save version 1
    v1_id = storage.save_prompt(prompt_v1)
    print(f"\nSaved version 1: {v1_id}")

    # Create improved version
    prompt_v2 = Prompt(
        template="""Answer the following question thoughtfully:

Question: {{ question }}

Please provide a clear and accurate answer.""",
        system_message="You are a knowledgeable and helpful assistant.",
        name="qa-prompt",
        metadata={"author": "developer", "version": "2.0"},
    )

    # Save version 2
    v2_id = storage.save_prompt(prompt_v2)
    print(f"Saved version 2: {v2_id}")

    # Create another iteration
    prompt_v3 = Prompt(
        template="""{{ question }}

Answer concisely and accurately. If uncertain, say so.""",
        system_message="You are a precise and concise assistant.",
        name="qa-prompt",
        metadata={"author": "developer", "version": "3.0"},
    )

    v3_id = storage.save_prompt(prompt_v3)
    print(f"Saved version 3: {v3_id}")

    # View history
    print("\n" + "=" * 40)
    print("Version History:")
    print("-" * 40)

    history = storage.history("qa-prompt")
    for entry in history:
        modified = datetime.fromtimestamp(float(entry["modified"]))
        print(f"  {entry['version']}")
        print(f"    Modified: {modified.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"    Size: {entry['size']} bytes")
        print()

    # Load a specific version
    print("=" * 40)
    print("Loading version 1:")
    print("-" * 40)
    loaded_v1 = storage.load_prompt("qa-prompt", v1_id)
    print(f"Template: {loaded_v1.template}")

    # Load latest version
    print("\n" + "=" * 40)
    print("Loading latest version:")
    print("-" * 40)
    latest = storage.load_prompt("qa-prompt")
    print(f"Template: {latest.template}")
    print(f"Metadata version: {latest.metadata.get('version', 'N/A')}")

    # List all prompts
    print("\n" + "=" * 40)
    print("All stored prompts:")
    print("-" * 40)
    all_prompts = storage.list_prompts()
    for name in all_prompts:
        versions = storage.history(name)
        print(f"  {name}: {len(versions)} version(s)")


if __name__ == "__main__":
    main()
