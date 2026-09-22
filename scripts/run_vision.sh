#!/usr/bin/env bash
#
# Convenience wrapper around scripts/run_vision_friend.py.
# Defaults to dry-run so it is safe to execute without extra flags.
#
# The friend/chat title used for matching is read automatically from the
# generated profile (friend_name field). Use --window-title if you need to
# override the WeChat window title regex.
#
# Usage:
#   ./scripts/run_vision.sh --chat-id "friend_001"
#
# Real send (requires explicit --send):
#   ./scripts/run_vision.sh --chat-id "friend_001" --send
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER="${SCRIPT_DIR}/run_vision_friend.py"
PROFILE_DIR="${PROJECT_DIR}/data/profiles"

CHAT_ID=""
WINDOW_TITLE=""
TICKS=""
POLL_INTERVAL=""
MODE="dry-run"

show_help() {
    cat <<'EOF'
Run the WeChat vision gateway for a single friend.

Required:
  --chat-id <id>          Stable chat id used for the profile

Optional:
  --window-title <regex>  Regex to locate the WeChat window
                          (default: friend_name from profile)
  --send                  Enable real auto-send (default is dry-run)
  --ticks <n>             Number of polling ticks (default: 3)
  --poll-interval <sec>   Seconds between polls (default: 2.0)
  -h, --help              Show this help

Examples:
  ./scripts/run_vision.sh --chat-id "alice_001"
  ./scripts/run_vision.sh --chat-id "alice_001" --window-title "Alice" --send
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --chat-id)
            CHAT_ID="$2"
            shift 2
            ;;
        --window-title)
            WINDOW_TITLE="$2"
            shift 2
            ;;
        --ticks)
            TICKS="$2"
            shift 2
            ;;
        --poll-interval)
            POLL_INTERVAL="$2"
            shift 2
            ;;
        --send)
            MODE="send"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            show_help >&2
            exit 1
            ;;
    esac
done

if [[ -z "${CHAT_ID}" ]]; then
    echo "Error: --chat-id is required." >&2
    show_help >&2
    exit 1
fi

PROFILE_PATH="${PROFILE_DIR}/${CHAT_ID}.json"
if [[ ! -f "${PROFILE_PATH}" ]]; then
    echo "Error: profile not found: ${PROFILE_PATH}" >&2
    echo "Import chat history first, e.g.:" >&2
    echo "  python scripts/import_chat_history.py ${CHAT_ID} chat_history.json" >&2
    exit 1
fi

if [[ ! -f "${RUNNER}" ]]; then
    echo "Error: runner script not found at ${RUNNER}" >&2
    exit 1
fi

if ! command -v python &>/dev/null && ! command -v python3 &>/dev/null; then
    echo "Error: python or python3 is required." >&2
    exit 1
fi

PYTHON_CMD="$(command -v python3 || command -v python)"

cd "${PROJECT_DIR}"

# Resolve the friend/chat title from the profile so the vision gateway can
# match the chat title inside the screenshot.
FRIEND_NAME="$(${PYTHON_CMD} - <<PY
import json
with open('${PROFILE_PATH}', encoding='utf-8') as f:
    print(json.load(f).get('friend_name', '${CHAT_ID}'))
PY
)"
# If the profile's friend_name is just punctuation/symbols (e.g. vision model
# garbage like "!!!"), fall back to the chat id and let the user override with
# --window-title if needed.
if ! "${PYTHON_CMD}" -c "import re, sys; sys.exit(0 if re.search(r'[\\w一-鿿]', sys.argv[1]) else 1)" "${FRIEND_NAME}" 2>/dev/null; then
    echo "Warning: profile friend_name '${FRIEND_NAME}' looks like garbage, falling back to '${CHAT_ID}'." >&2
    echo "Use --window-title to specify the real WeChat window title if needed." >&2
    FRIEND_NAME="${CHAT_ID}"
fi

echo "Using friend/chat title from profile: ${FRIEND_NAME}"

# --friend-name is passed internally to the Python runner; it tells the vision
# extractor which chat title to look for inside the screenshot. Users of this
# wrapper do not need to provide it.
ARGS=(
    "${RUNNER}"
    "--chat-id" "${CHAT_ID}"
    "--friend-name" "${FRIEND_NAME}"
)

if [[ "${MODE}" == "dry-run" ]]; then
    ARGS+=("--dry-run")
else
    ARGS+=("--confirm-send")
fi

if [[ -n "${WINDOW_TITLE}" ]]; then
    ARGS+=("--window-title" "${WINDOW_TITLE}")
fi

if [[ -n "${TICKS}" ]]; then
    ARGS+=("--ticks" "${TICKS}")
fi

if [[ -n "${POLL_INTERVAL}" ]]; then
    ARGS+=("--poll-interval" "${POLL_INTERVAL}")
fi

echo "Starting vision gateway (mode=${MODE}) for '${FRIEND_NAME}'..."
exec "${PYTHON_CMD}" "${ARGS[@]}"
