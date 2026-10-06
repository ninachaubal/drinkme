#!/usr/bin/env bash
# Greedy A/B smoke for a new model: serve compressed and --stock sequentially,
# same fixed prompt at temperature 0, and print the two transcripts' agreement:
# the same text, or where they part. Over HTTP there are no logits, so a part
# here cannot be told from a near-tie (docs/method.md, "Numerical behavior");
# bench/stock_token_identity.py prints the margin where the arms part.
# Usage: greedy_ab_smoke.sh <hf-repo> [extra serve env, e.g. DRINKME_PREFIX_SLOTS=0]
# The exit code is 1 when an arm fails (a server that dies during load, a
# failed request, a malformed body) and 0 otherwise, whether or not the
# transcripts agree (this script is bash — the torch exit-clobber only eats
# python's codes; see docs/dev-environment.md, Known behavior).
set -u
MODEL="${1:?usage: greedy_ab_smoke.sh <hf-repo>}"
PORT_C=3711
PORT_S=3712
UV="${UV:-uv}"
cd "$(dirname "$0")/.."

PROMPT='{"model":"'"$MODEL"'","messages":[{"role":"user","content":"List the first five prime numbers, comma-separated, nothing else."}],"temperature":0,"max_tokens":512}'

serve_and_ask() { # $1=arm-name $2=port $3=extra-args
  local out="/tmp/qsmoke_$1.json"
  DRINKME_PREFIX_SLOTS=0 $UV run --no-sync drinkme serve --model "$MODEL" --port "$2" $3 \
    > "/tmp/qsmoke_$1.serve.log" 2>&1 &
  local pid=$!
  for i in $(seq 1 240); do  # up to 20 min for the load
    curl -sf -m 2 "http://127.0.0.1:$2/health" >/dev/null 2>&1 && break
    kill -0 $pid 2>/dev/null || { echo "ARM $1: server died during load"; tail -5 "/tmp/qsmoke_$1.serve.log"; return 1; }
    sleep 5
  done
  curl -sf -m 600 "http://127.0.0.1:$2/v1/chat/completions" \
    -H 'Content-Type: application/json' -d "$PROMPT" > "$out"
  local rc=$?
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  [ $rc -eq 0 ] || { echo "ARM $1: completion request failed rc=$rc"; return 1; }
  python3 -c "import json,sys; d=json.load(open('$out')); print(d['choices'][0]['message'].get('content') or '')" > "/tmp/qsmoke_$1.txt" \
    || { echo "ARM $1: malformed completion body"; cat "$out"; return 1; }
  echo "ARM $1 ok ($(wc -c < /tmp/qsmoke_$1.txt) bytes)"
}

serve_and_ask compressed $PORT_C ""        || exit 1
serve_and_ask stock      $PORT_S "--stock" || exit 1

if cmp -s /tmp/qsmoke_compressed.txt /tmp/qsmoke_stock.txt; then
  echo "VERDICT: both arms answered; greedy transcripts agree ($(wc -c < /tmp/qsmoke_stock.txt) bytes)"
  echo "--- transcript head ---"; head -c 300 /tmp/qsmoke_stock.txt; echo
else
  python3 - <<'PY'
a = open("/tmp/qsmoke_compressed.txt").read()
b = open("/tmp/qsmoke_stock.txt").read()
at = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
print(f"VERDICT: both arms answered; greedy transcripts part at char {at} of {len(a)}/{len(b)} "
      "(compressed/stock); no logits over HTTP to read the margin there")
print(f"  compressed: {a[max(0, at - 40):at + 40]!r}")
print(f"  stock:      {b[max(0, at - 40):at + 40]!r}")
PY
fi
exit 0
