#!/usr/bin/env bash
# Hot-loop GPU A/B driver: old loop (DRINKME_HOTLOOP_OFF=1) vs
# new loop (DRINKME_DETOK_VERIFY=1, window checked every token) on the real
# packs, real tokenizers, real device. Three pairs:
#   8B  serial      — standard-attention family
#   27B serial      — hybrid family, the showcase model
#   27B DRINKME_SPEC=mtp DRINKME_MTP_DEPTH=4 — the changed pick_row + window under speculation
# One process per arm: DRINKME_HOTLOOP_OFF and the speculation env are read once per
# engine, so a per-request env flip does not take effect.
# Verdicts ride the output; bash carries real exit codes (the torch clobber
# eats python's).
set -uo pipefail
cd "$(dirname "$0")/.."
REPO="$(git rev-parse --show-toplevel)"
PY="${PY:-$REPO/.venv/bin/python}"
export PYTHONPATH=$PWD/src
R=/tmp/hotloop_ab
mkdir -p $R

run_pair() { # $1=tag $2=repo $3=pack $4=extra-env $5=revision
  local tag=$1 repo=$2 pack=$3 env=$4 rev=${5:-}
  echo "=== pair $tag ($env) ==="
  env DRINKME_HOTLOOP_OFF=1 $env $PY bench/hotloop_gpu_acceptance.py \
      "$repo" "$pack" "$R/${tag}_old.json" $rev 2>&1 | tail -20
  [ -s "$R/${tag}_old.json" ] || { echo "VERDICT $tag: OLD ARM FAILED"; return 1; }
  env DRINKME_DETOK_VERIFY=1 $env $PY bench/hotloop_gpu_acceptance.py \
      "$repo" "$pack" "$R/${tag}_new.json" $rev 2>&1 | tail -20
  [ -s "$R/${tag}_new.json" ] || { echo "VERDICT $tag: NEW ARM FAILED"; return 1; }
  $PY - "$R/${tag}_old.json" "$R/${tag}_new.json" "$tag" <<'EOF'
import json, sys
old, new, tag = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]
bad = []
for k in old["cells"]:
    o, n = old["cells"][k], new["cells"][k]
    same = o["text"] == n["text"]
    note = "" if same else " *** DIFFERS ***"
    if not same:
        bad.append(k)
        i = next((j for j in range(min(len(o["text"]), len(n["text"])))
                  if o["text"][j] != n["text"][j]), min(len(o["text"]), len(n["text"])))
        note += f" first-divergence at char {i}"
    print(f"  {tag}/{k}: old {o['tokens']}t @{o['decode_toks_per_s']}tok/s | "
          f"new {n['tokens']}t @{n['decode_toks_per_s']}tok/s"
          f"{' valid_json=' + str(n.get('valid_json')) if 'json' in k else ''}{note}")
print(f"VERDICT {tag}: " + ("ALL CELLS TOKEN-IDENTICAL" if not bad
      else f"{len(bad)} cells differ: {bad}"))
sys.exit(0 if not bad else 1)
EOF
}

rc=0
run_pair 8b  Qwen/Qwen3-8B   ~/.cache/drinkme/packs/Qwen--Qwen3-8B@b968826d9c46 "" b968826d9c46 || rc=1
run_pair 27b Qwen/Qwen3.8-27B ~/.cache/drinkme/packs/Qwen--Qwen3.8-27B@1d4bf0f2ff60 "" 1d4bf0f2ff60 || rc=1
run_pair 27b-mtp Qwen/Qwen3.8-27B ~/.cache/drinkme/packs/Qwen--Qwen3.8-27B@1d4bf0f2ff60 "DRINKME_SPEC=mtp DRINKME_MTP_DEPTH=4" 1d4bf0f2ff60 || rc=1
echo "=== FINAL: $([ $rc -eq 0 ] && echo ALL PAIRS PASS || echo FAILURES ABOVE) ==="
exit $rc
