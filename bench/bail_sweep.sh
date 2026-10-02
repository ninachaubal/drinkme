#!/usr/bin/env bash
# bail_sweep.sh — the adaptive bail's trigger, measured.
#
# The bail (serving/mtp.py) trips when acceptance over the last
# DRINKME_MTP_BAIL_WINDOW drafted positions falls below DRINKME_MTP_BAIL_FLOOR,
# and DRINKME_MTP_REARM tokens later drafts again. The default (32 / 0.15 / 64)
# tripped on 70% of ordinary sampled 256-token answers on Qwen3.8-27B
# (bench/mtp_ask.py, gfx1151), at 6-12% over a window whose denominator is
# drafted positions.
# This script serves each configuration in turn and drives bench/mtp_ask.py
# at it, so the trigger is set on evidence. Deterministic, no model in the
# loop: run it detached and read the table afterwards
# (bench/bail_sweep_table.py <out-dir>).
#
# Per config: gate MemAvailable >= 65 GB, serve the 27B sip pack on :3299
# with the config's env (the fixed serving env below), wait
# for /health, check the device, one warm-up, then mtp_ask.py x RUNS on each
# prompt (P1 free text — the prompt that trips; P2 a table of primes, hostile
# to the head, to price the probes), sampled at the server's defaults, 256
# tokens; then kill the server by PID and confirm :3299 is gone.
#
# Output: <out-dir>/<config-dir>/{P1,P2}.json (mtp_ask's records), P1.log /
# P2.log (its stdout), serve.log (the server's stderr), env.txt. A config
# whose JSONs already exist is skipped, so a rerun resumes. One line per
# config as it finishes; the whole log is also appended to <out-dir>/sweep.log.
#
# Config grammar: W<window>/F<floor>/R<rearm>, e.g. W64/F0.08/R64; the
# directory name is the same with '-' for '/'. --configs takes a comma list.
#
#   bench/bail_sweep.sh                       # the default list, 10 runs
#   bench/bail_sweep.sh --configs W32/F0.15/R64 --runs 1   # a smoke
#   bench/bail_sweep.sh --configs W64/F0.08/R16,W64/F0.08/R32
#   bench/bail_sweep.sh --p2 "<another prompt>" --out-dir verification/bail-sweep-x
#
# --p1 / --p2 replace a prompt's text (a primes table as P2 accepted at 94%
# in a smoke run — not hostile; a rerun with another P2 is a different
# out-dir, since the records are per prompt NAME).
#
# Takes /tmp/drinkme-gpu.lock (blocking) around the whole run; --no-lock says
# the caller holds it. NEVER touches :3215, drinkme serve's default port,
# where a working server may be listening: its own server is on --port
# (default 3299, and 3215 is refused), started and killed here by PID. The verdict is in the output,
# not in $? (AGENTS.md: TheRock's exit handler can mask a failure).
set -u

# --------------------------------------------------------------- arguments --
CONFIGS="W32/F0.15/R64,W64/F0.15/R64,W128/F0.15/R64,W32/F0.08/R64,W64/F0.08/R64,W128/F0.08/R64,W64/F0.08/R16,W64/F0.08/R32"
RUNS=10
MAX_TOKENS=256
PORT=3299
CTX=8192
MODEL=Qwen/Qwen3.8-27B
PACK="$HOME/.cache/drinkme/packs/Qwen--Qwen3.8-27B@1d4bf0f2ff60"
DATE=$(date +%F)
OUT=""
PROMPTS="P1,P2"
MEM_GATE_GB=65
MEM_WAIT_S=900
HEALTH_WAIT_S=600
LOCK=/tmp/drinkme-gpu.lock
TAKE_LOCK=1
PY=${PY:-.venv/bin/python}
P1_TEXT=""; P2_TEXT=""

usage() { sed -n '2,43p' "$0" | sed 's/^# \{0,1\}//'; }
while [ $# -gt 0 ]; do
  case "$1" in
    --configs) CONFIGS=$2; shift 2 ;;
    --runs) RUNS=$2; shift 2 ;;
    --max-tokens) MAX_TOKENS=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --ctx) CTX=$2; shift 2 ;;
    --model) MODEL=$2; shift 2 ;;
    --pack) PACK=$2; shift 2 ;;
    --date) DATE=$2; shift 2 ;;
    --out-dir) OUT=$2; shift 2 ;;
    --prompts) PROMPTS=$2; shift 2 ;;
    --mem-gate-gb) MEM_GATE_GB=$2; shift 2 ;;
    --python) PY=$2; shift 2 ;;
    --p1) P1_TEXT=$2; shift 2 ;;
    --p2) P2_TEXT=$2; shift 2 ;;
    --no-lock) TAKE_LOCK=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if [ "$PORT" = "3215" ]; then
  echo "REFUSING: :3215 is drinkme serve's default port, where a working server may be listening; pick another --port" >&2
  exit 2
fi

# ------------------------------------------------------------------ lock --
# The GPU lock, held on fd 9 for the whole run (blocking: a detached sweep
# waits its turn). The server child closes fd 9 so a server that outlived
# this script could not hold the lock with it.
if [ "$TAKE_LOCK" = 1 ]; then
  exec 9>>"$LOCK"
  echo "[$(date -u +%FT%TZ)] [sweep] waiting for $LOCK"
  flock 9
  echo "[$(date -u +%FT%TZ)] [sweep] holding $LOCK"
fi

cd "$(dirname "$0")/.." || exit 1
OUT=${OUT:-verification/bail-sweep-$DATE}
mkdir -p "$OUT"
exec > >(tee -a "$OUT/sweep.log") 2>&1

P1=${P1_TEXT:-"How do I kill a Python process that is hogging my GPU?"}
P2=${P2_TEXT:-"Print a table of the first 40 prime numbers with their squares and cubes, one per line, no commentary."}

log() { echo "[$(date -u +%FT%TZ)] [sweep] $*"; }
avail_gb() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
listeners() { ss -ltnp 2>/dev/null | grep -c ":$PORT "; }
prompt_text() { case "$1" in P1) printf '%s' "$P1" ;; P2) printf '%s' "$P2" ;; *) return 1 ;; esac; }

SPID=""
kill_server() {
  [ -n "$SPID" ] || return 0
  log "killing :$PORT server pid $SPID"
  kill "$SPID" 2>/dev/null
  for _ in $(seq 1 30); do kill -0 "$SPID" 2>/dev/null || break; sleep 1; done
  if kill -0 "$SPID" 2>/dev/null; then log "pid $SPID still up after 30s: SIGKILL"; kill -9 "$SPID" 2>/dev/null; fi
  wait "$SPID" 2>/dev/null
  SPID=""
  for _ in $(seq 1 30); do [ "$(listeners)" = 0 ] && break; sleep 1; done
  log "port check: $(listeners) listeners on :$PORT"
}
trap 'kill_server; log "end (MemAvailable $(avail_gb)G)"' EXIT
trap 'log "signalled: stopping"; exit 130' INT TERM  # so the EXIT trap (kill_server) runs under a timeout too

log "=== bail sweep: configs=$CONFIGS runs=$RUNS max_tokens=$MAX_TOKENS prompts=$PROMPTS out=$OUT"
log "P1: $P1"
log "P2: $P2"
log "tree $(git rev-parse --short HEAD 2>/dev/null) ($(git rev-parse --abbrev-ref HEAD 2>/dev/null)); python $PY; pack $PACK"
log "default port :3215 (never touched): $(curl -s -m 3 localhost:3215/health || echo 'not up')"

T_ALL=$(date +%s)
DONE=0; SKIPPED=0; FAILED=0
IFS=',' read -r -a CFGS <<< "$CONFIGS"
IFS=',' read -r -a PRS <<< "$PROMPTS"
for cfg in "${CFGS[@]}"; do
  if ! [[ "$cfg" =~ ^W([0-9]+)/F([0-9.]+)/R([0-9]+)$ ]]; then
    log "BAD CONFIG '$cfg' (grammar: W<window>/F<floor>/R<rearm>) — skipped"; FAILED=$((FAILED+1)); continue
  fi
  W=${BASH_REMATCH[1]}; F=${BASH_REMATCH[2]}; R=${BASH_REMATCH[3]}
  dir="$OUT/${cfg//\//-}"
  mkdir -p "$dir"
  todo=()
  for pr in "${PRS[@]}"; do [ -s "$dir/$pr.json" ] || todo+=("$pr"); done
  if [ ${#todo[@]} -eq 0 ]; then
    log "$cfg: every prompt's JSON exists — skipped (idempotent)"; SKIPPED=$((SKIPPED+1)); continue
  fi
  T0=$(date +%s)
  log "--- $cfg (window $W drafted, floor $F, rearm $R): prompts ${todo[*]}"

  # the memory gate: the model beside any server already resident needs ~65 GB free
  waited=0
  while [ "$(avail_gb)" -lt "$MEM_GATE_GB" ]; do
    if [ "$waited" -ge "$MEM_WAIT_S" ]; then break; fi
    [ "$waited" = 0 ] && log "MemAvailable $(avail_gb)G < ${MEM_GATE_GB}G: waiting (up to ${MEM_WAIT_S}s)"
    sleep 10; waited=$((waited+10))
  done
  if [ "$(avail_gb)" -lt "$MEM_GATE_GB" ]; then
    log "$cfg: SKIPPED — MemAvailable $(avail_gb)G < ${MEM_GATE_GB}G after ${MEM_WAIT_S}s"; FAILED=$((FAILED+1)); continue
  fi
  if [ "$(listeners)" != 0 ]; then
    log "$cfg: SKIPPED — something already listens on :$PORT"; FAILED=$((FAILED+1)); continue
  fi

  # serve: the receipts' fixed env + the config's three knobs
  : > "$dir/serve.log"
  {
    echo "DRINKME_MTP_BAIL_WINDOW=$W"; echo "DRINKME_MTP_BAIL_FLOOR=$F"; echo "DRINKME_MTP_REARM=$R"
    echo "DRINKME_PREFIX_SLOTS=1 DRINKME_MTP_DEPTH=4 TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DRINKME_NO_AUTO_DEPS=1 OMP_NUM_THREADS=8"
    echo "serve --model $MODEL --pack-dir $PACK --port $PORT --ctx $CTX"
    echo "tree $(git rev-parse --short HEAD 2>/dev/null); MemAvailable $(avail_gb)G at start"
  } > "$dir/env.txt"
  log "MemAvailable $(avail_gb)G; serving :$PORT"
  (
    exec 9>&-
    exec env DRINKME_MTP_BAIL_WINDOW="$W" DRINKME_MTP_BAIL_FLOOR="$F" DRINKME_MTP_REARM="$R" \
      DRINKME_PREFIX_SLOTS=1 DRINKME_MTP_DEPTH=4 TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DRINKME_NO_AUTO_DEPS=1 OMP_NUM_THREADS=8 \
      PYTHONUNBUFFERED=1 PYTHONPATH=src \
      "$PY" -m drinkme.cli serve --model "$MODEL" --pack-dir "$PACK" --port "$PORT" --ctx "$CTX"
  ) > "$dir/serve.log" 2>&1 &
  SPID=$!
  log "server pid $SPID, waiting for /health (up to ${HEALTH_WAIT_S}s)"
  ok=0
  for _ in $(seq 1 $((HEALTH_WAIT_S / 3))); do
    if grep -q "REFUSING" "$dir/serve.log"; then log "SERVER REFUSED:"; grep REFUSING "$dir/serve.log"; break; fi
    if ! kill -0 "$SPID" 2>/dev/null; then log "SERVER DIED:"; tail -20 "$dir/serve.log"; SPID=""; break; fi
    if curl -sf -m 3 "localhost:$PORT/health" >/dev/null; then ok=1; break; fi
    sleep 3
  done
  if [ "$ok" != 1 ]; then log "$cfg: FAILED — no /health"; kill_server; FAILED=$((FAILED+1)); continue; fi
  H=$(curl -s -m 3 "localhost:$PORT/health")
  DEV=$(grep -m1 "\[drinkme\] device:" "$dir/serve.log")
  log "health: $H"
  log "boot: $DEV"
  case "$H" in *'"device": "cuda:0"'*) ;; *) log "$cfg: FAILED — WRONG DEVICE (expected cuda:0); stopping the sweep"; kill_server; FAILED=$((FAILED+1)); break ;; esac
  case "$DEV" in *"device: cuda"*) ;; *) log "$cfg: FAILED — boot log device line is not cuda; stopping the sweep"; kill_server; FAILED=$((FAILED+1)); break ;; esac
  grep -E "\[drinkme\.mtp\] (head loaded|DRINKME_MTP_DEPTH)" "$dir/serve.log" | head -4

  # one warm-up (kernels, the prefix slot), then the runs
  curl -s -m 300 "localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello.\"}],\"max_tokens\":16}" >/dev/null
  summary=""
  for pr in "${todo[@]}"; do
    text=$(prompt_text "$pr") || { log "unknown prompt $pr"; continue; }
    log "$cfg $pr: mtp_ask x$RUNS"
    OMP_NUM_THREADS=8 PYTHONPATH=src "$PY" bench/mtp_ask.py --port "$PORT" --model "$MODEL" \
      --runs "$RUNS" --max-tokens "$MAX_TOKENS" --prompt "$text" --label "27b_sip_${cfg//\//-}_$pr" \
      --server-log "$dir/serve.log" --out "$dir/$pr.json" > "$dir/$pr.log" 2>&1
    v=$(grep -E '^\[ask\] (VERDICT|REFUSING|ERROR)' "$dir/$pr.log" | tail -1)
    s=$(grep -E '^\[ask\] line:' "$dir/$pr.log" | tail -1 | sed 's/^\[ask\] line: //')
    log "$cfg $pr: ${v:-no verdict} — ${s:-no summary}"
    summary="$summary$pr[${s:-none}] "
    # an incomplete record is not a receipt: a rerun does this prompt again
    case "$v" in *"VERDICT: OK"*) ;; *) rm -f "$dir/$pr.json" ;; esac
  done
  log "metrics: $(curl -s -m 3 "localhost:$PORT/metrics" | grep -E '^drinkme_mtp_(bail_trips|rearms)_total' | tr '\n' ' ')"
  kill_server
  T1=$(date +%s)
  echo "[$(date -u +%FT%TZ)] [sweep] DONE $cfg in $((T1 - T0))s: $summary"
  DONE=$((DONE+1))
done
log "=== sweep over: $DONE configs done, $SKIPPED skipped (existing), $FAILED failed, $(( $(date +%s) - T_ALL ))s total"
log "table: PYTHONPATH=src $PY bench/bail_sweep_table.py $OUT"
