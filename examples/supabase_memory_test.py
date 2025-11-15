"""Simple Supabase + mem0 memory smoke test.

Steps to use:
1. Copy `.env.example` to `.env` and fill in SUPABASE_DB_URL and OPENAI_API_KEY.
2. Ensure vecs and mem0 optional deps are installed, e.g. from repo root:
   pip install ".[vector_stores,llms]"
3. Run:
   python examples/supabase_memory_test.py
"""

import os
import sys
from typing import Any

try:
    # Optional: load variables from a local .env file if python-dotenv is installed
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    # It's fine if python-dotenv is not installed; we just rely on OS env vars
    pass

from mem0 import Memory
from mem0.configs.base import MemoryConfig


def build_memory_from_env() -> Memory:
    """Configure mem0 Memory to use Supabase (pgvector) as vector store."""

    supabase_url = os.getenv("SUPABASE_DB_URL")
    if not supabase_url or not supabase_url.startswith("postgresql://"):
        raise SystemExit(
            "SUPABASE_DB_URL is not set or is not a valid postgresql:// URL. "
            "Please set it in your environment or .env file."
        )

    collection_name = os.getenv("MEM0_COLLECTION_NAME", "mem0_memories")

    try:
        embedding_dims = int(os.getenv("MEM0_EMBEDDING_DIMS", "1536"))
    except ValueError:
        raise SystemExit("MEM0_EMBEDDING_DIMS must be an integer.")

    # OPENAI_API_KEY is read internally by mem0's OpenAI LLM and embedding classes
    if not os.getenv("OPENAI_API_KEY"):
        print("[WARN] OPENAI_API_KEY is not set; mem0 will not be able to call OpenAI.")

    config = MemoryConfig(
        # Use Supabase as the vector store
        vector_store={
            "provider": "supabase",
            "config": {
                "connection_string": supabase_url,
                "collection_name": collection_name,
                "embedding_model_dims": embedding_dims,
            },
        },
        # Keep default llm/embedder configs (OpenAI provider, reads OPENAI_API_KEY)
    )

    return Memory(config=config)


def run_test() -> None:
    """Run a simple add + search round-trip against Supabase-backed memory."""

    print("[mem0] Initializing Memory with Supabase vector store...")
    memory = build_memory_from_env()

    user_id = os.getenv("MEM0_TEST_USER_ID", "test-user")

    test_message = "My name is Alice and I am building a SaaS startup."
    print(f"[mem0] Adding memory for user_id={user_id!r}: {test_message}")

    add_result: dict[str, Any] = memory.add(test_message, user_id=user_id)
    print("[mem0] Add result:", add_result)

    query = "What is my name?"
    print(f"[mem0] Searching memories with query: {query!r}")
    search_result: dict[str, Any] = memory.search(query=query, user_id=user_id, limit=5)

    print("[mem0] Search raw result:")
    print(search_result)

    results = search_result.get("results", [])
    print("\n[mem0] Top memories:")
    for idx, item in enumerate(results, start=1):
        memory_text = item.get("memory") or item.get("payload", {}).get("data")
        score = item.get("score")
        print(f"  {idx}. score={score} memory={memory_text}")


if __name__ == "__main__":
    try:
        run_test()
    except SystemExit as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    except Exception as e:  # pragma: no cover - simple smoke test script
        print("[ERROR] Unexpected exception while running Supabase memory test:", e, file=sys.stderr)
        sys.exit(1)
