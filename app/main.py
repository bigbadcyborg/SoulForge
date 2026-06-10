"""SoulForge TUI entry point (Iteration 1: modular CLI chat loop).

Run with:

    python -m app.main

This preserves the original ``chatbot.py`` behavior (SOUL persona + RAG over a
local ChromaDB) but routes everything through configurable, single-purpose
modules. A real Textual TUI replaces this loop in Iteration 2.
"""

from __future__ import annotations

from app.core.config import PROJECT_ROOT, AppConfig, load_config
from app.core.model_runtime import ModelRuntime
from app.core.prompt_builder import PromptBuilder
from app.memory.memory_manager import MemoryManager
from app.rag.retriever import Retriever

SOUL_PATH = PROJECT_ROOT / "SOUL.md"
EXIT_COMMANDS = {"exit", "quit"}


def load_soul() -> str:
    """Load the persona from SOUL.md, or fall back to a neutral default."""
    if not SOUL_PATH.exists():
        return "You are a helpful local chatbot."
    return SOUL_PATH.read_text(encoding="utf-8", errors="ignore").strip()


def feature_summary(config: AppConfig) -> str:
    features = config.features
    flags = {
        "soul": features.soul,
        "rag": features.rag,
        "memory": features.memory,
        "streaming": features.streaming,
        "sources": features.show_sources,
    }
    enabled = [name for name, on in flags.items() if on]
    return ", ".join(enabled) if enabled else "none"


def run() -> None:
    config = load_config()

    runtime = ModelRuntime(config)
    prompt_builder = PromptBuilder(config)
    memory_manager = MemoryManager(config)
    retriever = Retriever(config, runtime) if config.features.rag else None

    soul_text = load_soul() if config.features.soul else ""
    memory_snapshot = memory_manager.load() if config.features.memory else None

    runtime.load_chat_model()
    if retriever is not None:
        runtime.load_embedding_model()

    system_prompt = prompt_builder.build_system_prompt(soul_text, memory_snapshot)
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    print(f"\nModel: {config.model.chat_model.name}")
    print(f"Active features: {feature_summary(config)}")
    print("Local RAG chatbot started. Type 'exit' to quit.")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in EXIT_COMMANDS:
            break

        context_text = ""
        if retriever is not None:
            chunks = retriever.retrieve(user_input)
            context_text = retriever.format_context(chunks)
            if config.features.show_sources and chunks:
                print("\nSources:")
                for chunk in chunks:
                    print(f"  - {chunk.source} (chunk {chunk.chunk_index})")

        user_turn = prompt_builder.build_user_turn(user_input, context_text)
        messages.append({"role": "user", "content": user_turn})

        if config.features.streaming:
            print("\nBot: ", end="", flush=True)
            stream = runtime.create_chat_completion(messages, stream=True)
            parts: list[str] = []
            for token in runtime.iter_stream_text(stream):
                parts.append(token)
                print(token, end="", flush=True)
            print()
            assistant_reply = "".join(parts).strip()
        else:
            response = runtime.create_chat_completion(messages, stream=False)
            assistant_reply = response["choices"][0]["message"]["content"].strip()
            print(f"\nBot: {assistant_reply}")

        messages.append({"role": "assistant", "content": assistant_reply})


def main() -> None:
    try:
        run()
    except FileNotFoundError as error:
        print(f"\nStartup error: {error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
