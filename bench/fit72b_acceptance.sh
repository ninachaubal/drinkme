#!/usr/bin/env bash
# bench/fit72b_acceptance.sh — take the Qwen2.5-72B fit point with `drinkme
# bench` on a ~122 GB unified box, under a MEMORY WATCHDOG.
#
# Why this exists: the bench's compressed arm streams
# (arms.load_compressed_streaming) — the pack's ~103 GB resident + one
# tensor of host transient, never the 145 GB bf16 model on the host — so
# `drinkme bench` can take the 72B fit point the site carries (Qwen2.5-72B
# sip, compressed_weights_gb 103.482). On a
# 122 GB box with a desktop resident that leaves a few GB; a global OOM on
# unified memory takes the desktop with it, so the
# bench's own two guards (bench._compressed_fit at detect time,
# arms.refuse_if_compressed_wont_fit against live MemAvailable right before
# the load) get a third, outside the process: this loop, which kills THE
# BENCH PID ONLY (never `pkill drinkme` — any other `drinkme serve`, on its
# default port 3215 or elsewhere, may be someone's working server) if
# MemAvailable falls under FLOOR_GB, and records the minimum
# it saw. An honest outcome here is one of three: a record; a refusal
# before any load (the fit check/guard said it does not fit — nothing is
# written, the message names the numbers); or a watchdog kill (no record,
# the log says at what MemAvailable).
#
# Run from the checkout root (or a worktree) whose code you are accepting:
#   bench/fit72b_acceptance.sh
# Knobs (env): PY (the interpreter; default .venv/bin/python of the main
# checkout — a worktree has no .venv), START_MIN_GB (refuse to start below
# this MemAvailable; default 112), FLOOR_GB (kill below this; default 3).
#
# Expected numbers on a Strix Halo (gfx1151; 124 GiB = 133.1 GB unified,
# MemTotal 128.8 GiB = 131.9 GB): charge 103.32 (the sip pack's
# residentBytes) + 2.49 (the embedding, the checkpoint's largest tensor)
# = 105.8 GB, x1.1 = 116.4 GB against 133.1
# — the fit check says fits. Live, with the desktop (~3 GB anonymous) and
# whatever /tmp (a tmpfs there: stale slot dirs from serve smoke runs can
# hold 20 GB) holds: expect the run's MINIMUM
# MemAvailable ≈ start − 109 GB (103.3 resident + ~2.5 torch/HIP runtime +
# ~1 KV/activations/logits + the 2.49 GB embedding transient at the very
# end of the load, when everything else is already resident; the packer's
# ~10x scratch lands earlier, largest tensor first, while little is
# resident). So START_MIN_GB=112 -> a floor of ~3 GB; if MemAvailable at
# start is under that, free the box first (close the browser, `rm -rf` the
# stale /tmp/drinkme-*-slots dirs, no other model on the GPU) rather than
# lowering the floor.
set -u

MODEL=${MODEL:-Qwen2.5-72B}
PY=${PY:-.venv/bin/python}
START_MIN_GB=${START_MIN_GB:-112}
FLOOR_GB=${FLOOR_GB:-3}
STAMP=$(date +%Y%m%d-%H%M%S)
mkdir -p measurements
BENCHLOG=measurements/fit72b-bench-$STAMP.log
WATCHLOG=measurements/fit72b-watchdog-$STAMP.log

mem_avail_gb() { awk '/MemAvailable/ {printf "%.2f", $2 / 1e6}' /proc/meminfo; }
lt() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a < b) }'; }  # float a < b, no bc needed

echo "== 0. never uv sync; interpreter: $PY"
PYTHONPATH=src "$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || exit 2

echo "== 1. the snapshot must be on the box (bench downloads at load time otherwise, 145 GB, inside the run)"
SNAP=$(PYTHONPATH=src "$PY" - <<'EOF'
from drinkme.serving.checkpoint import cached_snapshot_dir
from drinkme.suggest import MODELS
import os
m = next(x for x in MODELS if x.name == os.environ.get("MODEL", "Qwen2.5-72B"))
print(cached_snapshot_dir(m.hf_repo, m.revision) or "")
EOF
)
if [ -z "$SNAP" ]; then
  echo "!! $MODEL is not cached (config/index only, or nothing). Download it first (145 GB; ~1.5 TB free on /):"
  echo "   $(dirname "$PY")/hf download Qwen/Qwen2.5-72B-Instruct --revision 495f39366efef23836d0cfae4fbe635880d2be31"
  echo "   (the venv's own hf CLI; then re-run this script)"
  exit 2
fi
echo "   snapshot: $SNAP"

echo "== 2. the plan (no torch, no load): must be arms [compressed], charged ~103 GB + one tensor"
PLANFILE=measurements/fit72b-plan-$STAMP.json
PYTHONPATH=src "$PY" -m drinkme.cli bench --dry-run --model "$MODEL" > "$PLANFILE" || exit 2
cat "$PLANFILE"
"$PY" - "$PLANFILE" <<'EOF' || exit 2
import json, sys
p = json.load(open(sys.argv[1]))["points"][0]
assert p["arms"] == ["compressed"], p["arms"]
assert p["stockArmPredictedToFit"] is False and p["twinArmPredictedToFit"] is False
if not p["compressedArmPredictedToFit"]:
    if p["compressedArmFitsWithoutHeadroom"]:
        print("!! knife-edge: the compressed arm fits only WITHOUT the x1.1 headroom — an OOM is the honest result")
    else:
        print("!! the compressed arm does not fit this box even without headroom — bench will refuse; no record"); sys.exit(2)
print(f"   plan ok: compressed arm {p['compressedArmChargeGB']} GB charged (largest tensor "
      f"{p['largestTensorGB']} GB) x{p['compressedArmHeadroom']} against {p['stockArmCeilingGB']} GB")
EOF

echo "== 3. live memory before starting"
AVAIL=$(mem_avail_gb)
echo "   MemAvailable $AVAIL GB (need >= $START_MIN_GB GB to start; /tmp is a tmpfs on this box — stale /tmp/drinkme-*-slots dirs count against it)"
if lt "$AVAIL" "$START_MIN_GB"; then
  echo "!! not enough live memory — free the box (browser, /tmp slot dirs, other GPU users) and re-run; not lowering the floor for you"
  exit 2
fi

echo "== 4. bench under the watchdog (kills PID of THIS bench only, below $FLOOR_GB GB MemAvailable)"
echo "   bench log: $BENCHLOG · watchdog log: $WATCHLOG"
PYTHONPATH=src "$PY" -m drinkme.cli bench --model "$MODEL" > "$BENCHLOG" 2>&1 &
PID=$!
MIN=$AVAIL
KILLED=0
while kill -0 "$PID" 2>/dev/null; do
  NOW=$(mem_avail_gb)
  if lt "$NOW" "$MIN"; then MIN=$NOW; fi
  echo "$(date +%T) MemAvailable $NOW GB (min $MIN)" >> "$WATCHLOG"
  if lt "$NOW" "$FLOOR_GB"; then
    echo "$(date +%T) !! MemAvailable $NOW GB < floor $FLOOR_GB GB — killing bench pid $PID" | tee -a "$WATCHLOG"
    kill -TERM "$PID" 2>/dev/null; sleep 5; kill -KILL "$PID" 2>/dev/null
    KILLED=1
  fi
  sleep 1
done
wait "$PID" 2>/dev/null

echo "== 5. verdict (read the LOG, never \$?: on TheRock ROCm torch an atexit handler _exit(0)s over the real status)"
echo "   minimum MemAvailable during the run: $MIN GB (log: $WATCHLOG)"
if [ "$KILLED" = 1 ]; then
  echo "!! WATCHDOG KILLED the bench — no record; the honest result is: the compressed arm did not fit what was live ($MIN GB min). Free memory and re-run, or accept that this box cannot take the point."
  exit 3
fi
grep -E "REFUSING|fit check|compressed [0-9]|-> (\./)?measurements/|Error|error" "$BENCHLOG" | tail -n 20
# bench prints the path as `-> ./measurements/…` since the record-path change; accept both
if grep -qE "^-> (\./)?measurements/" "$BENCHLOG"; then
  REC=$(grep -E "^-> (\./)?measurements/" "$BENCHLOG" | tail -1 | cut -d' ' -f2)
  echo "   record: $REC"
  "$PY" - "$REC" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
print("   stock", r["stock"], "· raw.twin_outcome", r["raw"].get("twin_outcome"))
print("   metrics:", {m["name"]: m["value"] for m in r["metrics"]})
print("   verified:", r["raw"]["verified_tensors"], "/", r["raw"]["swapped_linears"], "·", r["compression"])
EOF
else
  echo "!! no record was written — see $BENCHLOG (a REFUSING line is the honest fit check/guard verdict, not a bug)"
  exit 3
fi
