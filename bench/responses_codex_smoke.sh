#!/usr/bin/env bash
# Codex CLI against a RUNNING drinkme server over /v1/responses — the second
# real client of the dialect (bench/responses_sdk_smoke.mjs is the first).
# Codex speaks ONLY the Responses API (its config reference, 2026-09:
# `wire_api = "responses"` "is the only supported value"), so this is the
# client the dialect exists for.
#
#   npm i -g @openai/codex                    # once
#   bench/responses_codex_smoke.sh --base-url http://127.0.0.1:3216/v1 \
#       -o measurements/responses_codex_smoke_<model>.json
#
# What it does: writes a throwaway CODEX_HOME with the config.toml below
# (your ~/.codex is never touched), makes a scratch git repo to run in (Codex
# refuses to run outside one), and runs ONE `codex exec --json` turn whose
# prompt can only be answered by calling a tool (Codex's own shell tool —
# a function tool on the wire, exactly what a user's session sends). The
# JSONL event stream and the last message are kept; the verdicts are:
#
#   turn_completed     the stream ended with a completed turn (no error event)
#   tool_called        a command_execution item ran (a function_call went out
#                      and its function_call_output came back — the round trip)
#   marker_in_answer   the final message repeats the string the command printed
#
# Exit 0 when all three hold, 1 otherwise, 2 when codex is not installed.
# --dry-run writes the config and prints the command without running codex.
#
# The config.toml this writes (the snippet for a real ~/.codex/config.toml):
#
#   model = "<id from GET /v1/models>"
#   model_provider = "drinkme"
#   model_context_window = <drinkme.contextWindow from /v1/models>
#   model_reasoning_effort = "medium"     # the 27B's template takes xhigh|medium|low ONLY
#   web_search = "disabled"               # Codex's default ("cached") may offer a hosted
#                                         # web_search tool; drinkme refuses hosted tools
#
#   [model_providers.drinkme]
#   name = "drinkme"
#   base_url = "http://127.0.0.1:3215/v1"
#   wire_api = "responses"
#   stream_idle_timeout_ms = 1800000      # a long prefill on a slow box must not read as "stuck"
#   # env_key = "DRINKME_API_KEY"         # only with `drinkme serve --auth`
#
# Notes for the GPU window: `store`, `include` and `prompt_cache_key` in
# Codex's requests are accepted and ignored (docs/serve-responses.md);
# reasoning comes back as a reasoning item's summary text, which Codex
# shows as the agent's thinking.

set -euo pipefail

BASE_URL="http://127.0.0.1:3215/v1"
MODEL=""
OUT=""
DRY_RUN=0
EFFORT="medium"
PROMPT='Run exactly this shell command and then reply with its output verbatim and nothing else: echo drinkme-codex-roundtrip-ok'

usage() { sed -n '2,46p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --base-url) BASE_URL="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --effort) EFFORT="$2"; shift 2 ;;
    -o|--out) OUT="$2"; shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[codex-smoke] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
BASE_URL="${BASE_URL%/}"

if ! command -v codex >/dev/null 2>&1 && [ "$DRY_RUN" = 0 ]; then
  echo "[codex-smoke] codex is not on PATH. Install once: npm i -g @openai/codex" >&2
  exit 2
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "[codex-smoke] python3 is required (to read /v1/models and write the receipt)" >&2
  exit 2
fi

# the model id and its window, from the server itself
MODELS_JSON="$(curl -sf "$BASE_URL/models")" || { echo "[codex-smoke] GET $BASE_URL/models failed — is the server up?" >&2; exit 1; }
if [ -z "$MODEL" ]; then
  MODEL="$(printf '%s' "$MODELS_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"
fi
CTX="$(printf '%s' "$MODELS_JSON" | python3 -c '
import json, sys
d = json.load(sys.stdin)["data"]
m = next((x for x in d if x["id"] == sys.argv[1]), d[0])
print(m["drinkme"].get("contextWindow") or "")' "$MODEL")"

WORK="$(mktemp -d -t drinkme-codex-smoke.XXXXXX)"
export CODEX_HOME="$WORK/codex-home"
REPO="$WORK/repo"
mkdir -p "$CODEX_HOME" "$REPO"
{
  echo "model = \"$MODEL\""
  echo 'model_provider = "drinkme"'
  [ -n "$CTX" ] && echo "model_context_window = $CTX"
  echo "model_reasoning_effort = \"$EFFORT\""
  echo 'web_search = "disabled"'
  echo
  echo '[model_providers.drinkme]'
  echo 'name = "drinkme"'
  echo "base_url = \"$BASE_URL\""
  echo 'wire_api = "responses"'
  echo 'stream_idle_timeout_ms = 1800000'
  echo 'request_max_retries = 0'
  echo 'stream_max_retries = 0'
} > "$CODEX_HOME/config.toml"
( cd "$REPO" && git init -q && git -c user.name=smoke -c user.email=smoke@drinkme commit -q --allow-empty -m init )

echo "[codex-smoke] model $MODEL · base_url $BASE_URL · CODEX_HOME $CODEX_HOME"
echo "[codex-smoke] config.toml:"
sed 's/^/    /' "$CODEX_HOME/config.toml"

JSONL="$WORK/events.jsonl"
LAST="$WORK/last_message.txt"
CMD=(codex exec --json --skip-git-repo-check -C "$REPO" -o "$LAST" "$PROMPT")
echo "[codex-smoke] running: ${CMD[*]}"
if [ "$DRY_RUN" = 1 ]; then
  echo "[codex-smoke] dry run — not executing codex (config written above)"
  exit 0
fi

set +e
"${CMD[@]}" > "$JSONL" 2> "$WORK/stderr.txt"
RC=$?
set -e
echo "[codex-smoke] codex exit $RC · events → $JSONL"

# verdicts + receipt
python3 - "$JSONL" "$LAST" "$WORK/stderr.txt" "$RC" "$MODEL" "$BASE_URL" "$CODEX_HOME/config.toml" "$OUT" <<'EOF'
import json, os, sys, time
jsonl, last, stderr, rc, model, base, cfg, out = sys.argv[1:9]
events = []
for line in open(jsonl, errors="replace"):
    line = line.strip()
    if not line:
        continue
    try:
        events.append(json.loads(line))
    except json.JSONDecodeError:
        events.append({"type": "unparsed", "raw": line})
last_text = open(last, errors="replace").read() if os.path.exists(last) else ""
types = [e.get("type") for e in events]
items = [e.get("item", {}) for e in events if isinstance(e.get("item"), dict)]
tool_items = [i for i in items if i.get("type") in ("command_execution", "function_call", "mcp_tool_call")]
errors = [e for e in events if e.get("type") == "error"]
verdicts = {
    "turn_completed": int(rc) == 0 and "turn.completed" in types and not errors,
    "tool_called": bool(tool_items),
    "marker_in_answer": "drinkme-codex-roundtrip-ok" in last_text,
}
for k, v in verdicts.items():
    print(f"[codex-smoke] {k}: {'OK' if v else 'FAIL'}")
if errors:
    print(f"[codex-smoke] error events: {errors[:3]}")
if tool_items:
    t = tool_items[0]
    print(f"[codex-smoke] first tool item: {t.get('type')} {t.get('command') or t.get('name') or ''!s:.80}")
print(f"[codex-smoke] last message: {last_text.strip()[:200]!r}")
receipt = {"base_url": base, "client": "codex exec --json", "model": model,
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "config_toml": open(cfg).read(), "codex_exit": int(rc),
           "event_types": types, "events": events, "last_message": last_text,
           "stderr": open(stderr, errors="replace").read()[-4000:],
           "verdicts": verdicts}
if out:
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(receipt, f, indent=1)
    print(f"[codex-smoke] receipt → {out}")
ok = all(verdicts.values())
print("[codex-smoke] " + ("ALL HOLD" if ok else "FAILED: " + ", ".join(k for k, v in verdicts.items() if not v)))
sys.exit(0 if ok else 1)
EOF
