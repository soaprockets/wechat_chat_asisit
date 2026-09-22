#!/usr/bin/env bash
#
# End-to-end WeChat vision agent runner.
#
# This script wires together the full pipeline:
#   1. Check environment and optional dependencies
#   2. Import chat history and build a profile (if --chat-file is given)
#   3. Read the friend/chat title from the generated profile
#   4. Run the vision gateway (dry-run by default, real send with --send)
#
# The friend/chat title used for matching is read automatically from the
# generated profile (friend_name field). Use --window-title if you need to
# override the WeChat window title regex.
#
# Usage:
#   ./scripts/run_e2e.sh \
#       --chat-id "friend_001" \
#       --chat-file ~/Downloads/chat_history.json
#
# Real auto-send (after dry-run looks good):
#   ./scripts/run_e2e.sh \
#       --chat-id "friend_001" \
#       --chat-file ~/Downloads/chat_history.json \
#       --send
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMPORTER="${SCRIPT_DIR}/import_chat_history.py"
RUNNER="${SCRIPT_DIR}/run_vision_friend.py"
PROFILE_DIR="${PROJECT_DIR}/data/profiles"

CHAT_ID=""
CHAT_FILE=""
WINDOW_TITLE=""
TICKS=""
POLL_INTERVAL=""
MODE="dry-run"
AUTO_INSTALL=0

show_help() {
    cat <<'EOF'
End-to-end runner: import chat history, build profile, then run vision gateway.

Required:
  --chat-id <id>          Stable chat id used for the profile

Optional:
  --chat-file <path>      Chat history file to import before running
  --window-title <regex>  Regex to locate the WeChat window
                          (default: friend_name from profile)
  --send                  Enable real auto-send (default is dry-run)
  --ticks <n>             Number of polling ticks (default: 3)
  --poll-interval <sec>   Seconds between polls (default: 2.0)
  --install               Auto pip install -e ".[vision]" if deps missing
  -h, --help              Show this help

Examples:
  ./scripts/run_e2e.sh --chat-id "alice_001" --chat-file chat.json
  ./scripts/run_e2e.sh --chat-id "alice_001" --send
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --chat-id)
            CHAT_ID="$2"
            shift 2
            ;;
        --chat-file)
            CHAT_FILE="$2"
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
        --install)
            AUTO_INSTALL=1
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

if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
    echo "Error: python3 or python is required." >&2
    exit 1
fi

PYTHON_CMD="$(command -v python3 || command -v python)"

cd "${PROJECT_DIR}"

# Ensure .env exists with a template if the user has not configured it yet.
if [[ ! -f ".env" ]]; then
    if [[ -f ".env.example" ]]; then
        cp .env.example .env
        echo "Copied .env.example -> .env"
    fi
    echo ""
    echo "⚠️  Please edit .env and set LLM_BASE_URL, LLM_API_KEY and LLM_MODEL."
    echo "   Then run this script again."
    exit 1
fi

# Optionally install vision dependencies automatically.
if [[ "${AUTO_INSTALL}" -eq 1 ]]; then
    echo "Installing/updating project dependencies..."
    "${PYTHON_CMD}" -m pip install -e ".[vision]"
fi

# Verify that the optional vision dependencies are importable.
if ! "${PYTHON_CMD}" -c "import PIL, Quartz" 2>/dev/null; then
    echo ""
    echo "⚠️  Vision dependencies not found (Pillow + pyobjc-framework-Quartz)."
    echo "   Run one of:"
    echo "     pip install -e \".[vision]\""
    echo "     ./scripts/run_e2e.sh ... --install"
    exit 1
fi

# Import chat history if a file was provided.
if [[ -n "${CHAT_FILE}" ]]; then
    if [[ ! -f "${CHAT_FILE}" ]]; then
        echo "Error: chat file not found: ${CHAT_FILE}" >&2
        exit 1
    fi
    echo ""
    echo "==> Importing chat history for '${CHAT_ID}'..."
    "${PYTHON_CMD}" "${IMPORTER}" "${CHAT_ID}" "${CHAT_FILE}"
fi

# Confirm the profile exists.
PROFILE_PATH="${PROFILE_DIR}/${CHAT_ID}.json"
if [[ ! -f "${PROFILE_PATH}" ]]; then
    echo ""
    echo "Error: profile not found: ${PROFILE_PATH}" >&2
    echo "Provide --chat-file to import history first, or run import_chat_history.py manually." >&2
    exit 1
fi

# Resolve the friend/chat title from the profile so the vision gateway can
# match the chat title inside the screenshot.
FRIEND_NAME="$(${PYTHON_CMD} - <<PY
import json
with open('${PROFILE_PATH}', encoding='utf-8') as f:
    print(json.load(f).get('friend_name', '${CHAT_ID}'))
PY
    )"

echo ""
echo "==> Profile ready: ${PROFILE_PATH}"
echo "==> Using friend/chat title from profile: ${FRIEND_NAME}"
echo "==> Starting vision gateway (mode=${MODE}) for '${FRIEND_NAME}'..."
echo ""

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

exec "${PYTHON_CMD}" "${ARGS[@]}"
