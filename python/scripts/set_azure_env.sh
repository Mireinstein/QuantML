#!/usr/bin/env bash
# Paste the JSON output of `az ad sp create-for-rbac ...` into this script
# and it writes the matching values into python/.env for you -- no manual
# copy-paste of individual fields, and it never gets sent anywhere (this
# runs entirely on your machine; the values only ever touch your local
# .env file, which is gitignored).
#
# Usage:
#   cd python && ./scripts/set_azure_env.sh
#   (paste the JSON block, then press Ctrl-D)
#
#   AZURE_SUBSCRIPTION_ID=xxxx ./scripts/set_azure_env.sh   # skip the subscription-id lookup/prompt
#
# Writes/updates exactly four keys in python/.env, matching the names
# Terraform's azurerm provider reads automatically -- ARM_CLIENT_ID (appId),
# ARM_CLIENT_SECRET (password), ARM_TENANT_ID (tenant), and
# ARM_SUBSCRIPTION_ID (from $AZURE_SUBSCRIPTION_ID if set, else auto-detected
# via a local `az account show` if the Azure CLI is installed here,
# otherwise you'll be prompted for it -- a subscription ID isn't sensitive,
# it's just an identifier). Any other lines already in .env (e.g. the
# Alpaca paper-trading keys) are left alone.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_DIR="$(cd "$HERE/.." && pwd)"
ENV_FILE="$PYTHON_DIR/.env"

echo "Paste the JSON output of 'az ad sp create-for-rbac ...' below, then press Ctrl-D:"
JSON="$(cat)"

PARSED="$(python3 -c "
import json, sys
d = json.load(sys.stdin)
print(d['appId'])
print(d['password'])
print(d['tenant'])
" <<< "$JSON")"

APP_ID="$(sed -n '1p' <<< "$PARSED")"
CLIENT_SECRET="$(sed -n '2p' <<< "$PARSED")"
TENANT_ID="$(sed -n '3p' <<< "$PARSED")"

if [ -z "$APP_ID" ] || [ -z "$CLIENT_SECRET" ] || [ -z "$TENANT_ID" ]; then
  echo "Could not parse appId/password/tenant from that input -- did you paste the full JSON block?" >&2
  exit 1
fi

# Checked in this order: an explicit env var (also handy for non-interactive
# use), then a local Azure CLI if installed, then an interactive prompt as
# the last resort -- reading from /dev/tty specifically rather than stdin,
# since stdin was already consumed by the JSON paste above and reading from
# it again here would just get nothing.
SUBSCRIPTION_ID="${AZURE_SUBSCRIPTION_ID:-}"
if [ -z "$SUBSCRIPTION_ID" ]; then
  SUBSCRIPTION_ID="$(az account show --query id -o tsv 2>/dev/null || true)"
fi
if [ -z "$SUBSCRIPTION_ID" ]; then
  read -r -p "Azure CLI not available locally -- paste your subscription ID: " SUBSCRIPTION_ID < /dev/tty
fi

if [ ! -f "$ENV_FILE" ]; then
  cp "$PYTHON_DIR/.env.example" "$ENV_FILE"
fi

set_env_var() {
  local key="$1" value="$2"
  python3 - "$ENV_FILE" "$key" "$value" <<'PY'
import sys
path, key, value = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    with open(path) as f:
        lines = f.readlines()
except FileNotFoundError:
    lines = []
lines = [l for l in lines if not l.startswith(key + "=")]
lines.append(f"{key}={value}\n")
with open(path, "w") as f:
    f.writelines(lines)
PY
}

set_env_var "ARM_CLIENT_ID" "$APP_ID"
set_env_var "ARM_CLIENT_SECRET" "$CLIENT_SECRET"
set_env_var "ARM_TENANT_ID" "$TENANT_ID"
set_env_var "ARM_SUBSCRIPTION_ID" "$SUBSCRIPTION_ID"

echo "Wrote ARM_CLIENT_ID, ARM_CLIENT_SECRET, ARM_TENANT_ID, ARM_SUBSCRIPTION_ID to $ENV_FILE"
