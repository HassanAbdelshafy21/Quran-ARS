#!/usr/bin/env bash
# Scale a RunPod Serverless endpoint's ACTIVE workers up or down.
#
# RunPod has no built-in scheduler, so we call its REST API from cron.
# "workersMin" is the field behind the console's "Active workers" setting:
# workersMin=1 keeps one worker permanently warm (no cold starts, billed at the
# cheaper active rate); workersMin=0 lets the endpoint scale to zero (pay only
# per request, but the first request after idle pays a 1-3 min cold start).
#
# Usage:
#   ./runpod_scale.sh 1     # warm  (e.g. 10:00 — start of the working day)
#   ./runpod_scale.sh 0     # cold  (e.g. 17:00 — end of the working day)
#
# Required environment variables:
#   RUNPOD_API_KEY   - from runpod.io -> Settings -> API Keys
#   RUNPOD_ENDPOINT_ID - the serverless endpoint id (from its URL)
set -euo pipefail

WORKERS_MIN="${1:-}"
if [[ ! "$WORKERS_MIN" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 <workersMin>   e.g. $0 1  (warm)   |   $0 0  (scale to zero)" >&2
  exit 2
fi
: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY}"
: "${RUNPOD_ENDPOINT_ID:?set RUNPOD_ENDPOINT_ID}"

resp=$(curl -sS -w '\n%{http_code}' -X POST \
  "https://rest.runpod.io/v1/endpoints/${RUNPOD_ENDPOINT_ID}/update" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{\"workersMin\": ${WORKERS_MIN}}")

code=$(tail -n1 <<<"$resp")
body=$(sed '$d' <<<"$resp")

if [[ "$code" == "200" ]]; then
  echo "$(date -Is)  OK  workersMin=${WORKERS_MIN} on ${RUNPOD_ENDPOINT_ID}"
else
  echo "$(date -Is)  FAILED (HTTP ${code}) workersMin=${WORKERS_MIN}: ${body}" >&2
  exit 1
fi
