"""Import chat history from a user-provided file and build a profile.

The resulting profile is automatically persisted to the configured profile
store (default: ``./data/profiles/<chat_id>.json``).

Supports text (.txt), JSON (.json/.jsonl), CSV (.csv), PDF (.pdf) and images
(.png/.jpg/.jpeg). Images are sent directly to a vision-capable LLM; PDF parsing
requires the optional ``pymupdf`` dependency.

Usage:
    python scripts/import_chat_history.py friend_001 data/sample_chat.json
    python scripts/import_chat_history.py friend_001 chat_export.txt
    python scripts/import_chat_history.py friend_001 screenshot.png
    python scripts/import_chat_history.py friend_001 screenshot.png --friend-name "昵称"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wechat_agent.config import settings
from wechat_agent.ingestion import ChatHistoryImporter
from wechat_agent.memory.profile_store import JsonProfileStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import chat history from a file and build a profile"
    )
    parser.add_argument("chat_id", help="Unique chat ID (e.g., friend_001)")
    parser.add_argument("file", help="Path to the chat history file")
    parser.add_argument(
        "--friend-name",
        default=None,
        help="Override the friend's display name stored in the profile",
    )
    args = parser.parse_args()

    importer = ChatHistoryImporter()
    profile = importer.import_file(args.file, args.chat_id)

    store = JsonProfileStore(base_path=settings.profile_store_path)
    if args.friend_name:
        profile["friend_name"] = args.friend_name
        store.save_profile(args.chat_id, profile)

    print(json.dumps(profile, ensure_ascii=False, indent=2))

    profile_path = Path(settings.profile_store_path) / f"{args.chat_id}.json"
    print(f"\nProfile auto-saved to {profile_path}")


if __name__ == "__main__":
    main()
