#!/usr/bin/env bash
# The two-arm run for bench/mtp_sampled_dist.py: one server per arm, the same
# pack, one after the other, each on $PORT (default 3298; drinkme serve's
# default 3215 is refused, since a working server may be listening there).
#
# Arm 1 (mtp) serves the pack as shipped. Arm 2 (serial) needs
# DRINKME_SPEC=off, which is an engine-boot decision, so it is a second
# server. Each is killed by PID when its arm is done.
#
# The 27B needs the device to itself. If a resident server holds it, set
# SERVICE to its systemd user unit: the script stops that unit for the run
# and starts it again afterwards — on ANY exit, via the trap, so a dying
# script cannot leave it down. Unset, no service is touched.
#
#   PACK=~/.cache/drinkme/packs/Qwen--Qwen3.8-27B@1d4bf0f2ff60 N=400 bench/mtp_sampled_dist_run.sh
#   SERVICE=drinkme-server.service bench/mtp_sampled_dist_run.sh
set -eo pipefail
export PATH="$HOME/.local/bin:$HOME/.bun/bin:$PATH"
cd "$(dirname "$0")/.."
N="${N:-400}"
PORT="${PORT:-3298}"
OUT="${OUT:-verification/mtp_sampled_dist}"
PACK=${DRINKME_PACK:-$HOME/.cache/drinkme/packs/Qwen--Qwen3.8-27B@1d4bf0f2ff60}
SERVICE="${SERVICE:-}"
if [ "$PORT" = "3215" ]; then
  echo "REFUSING: :3215 is drinkme serve's default port, where a working server may be listening; set PORT" >&2
  exit 2
fi
mkdir -p "$OUT"
SP=""
restore() {
  if [ -n "$SP" ]; then kill "$SP" 2>/dev/null || true; wait "$SP" 2>/dev/null || true; fi
  if [ -n "$SERVICE" ]; then systemctl --user start "$SERVICE" || true; fi
}
trap restore EXIT

wait_health() {  # url, seconds
  for _ in $(seq 1 "$(( $2 / 3 ))"); do
    curl -sf -m 3 "$1/health" >/dev/null 2>&1 && return 0
    sleep 3
  done
  echo "no /health at $1 within $2s" >&2; return 1
}

arm() {  # name, extra env ("" for none)
  echo "== arm $1 (N=$N) $(date -Is)"
  env $2 TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 PYTHONUNBUFFERED=1 \
    uv run --no-sync drinkme serve --model Qwen/Qwen3.8-27B --ctx 8192 --pack-dir "$PACK" --port "$PORT" \
    > "$OUT/$1-server.log" 2>&1 &
  SP=$!
  wait_health "http://127.0.0.1:$PORT" 420
  grep -iE 'mtp|device:' "$OUT/$1-server.log" | head -5
  python3 bench/mtp_sampled_dist.py --base "http://127.0.0.1:$PORT" --arm "$1" --n "$N" --out "$OUT/arm-$1.json"
  kill "$SP"; wait "$SP" || true; SP=""
}

if [ -n "$SERVICE" ]; then
  echo "== stopping $SERVICE for the run $(date -Is)"
  systemctl --user stop "$SERVICE"
fi
arm mtp ""
arm serial "DRINKME_SPEC=off"
if [ -n "$SERVICE" ]; then
  echo "== $SERVICE back $(date -Is)"
  systemctl --user start "$SERVICE"
fi

echo "== compare"
python3 bench/mtp_sampled_dist.py --compare "$OUT/arm-mtp.json" "$OUT/arm-serial.json" | tee "$OUT/compare.txt"
echo "== done $(date -Is)"
