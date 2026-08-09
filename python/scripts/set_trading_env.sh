#!/usr/bin/env bash
# Reads ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY, DASHBOARD_USERNAME, and
# DASHBOARD_PASSWORD from python/.env and exports them as TF_VAR_* for
# this shell session, so `terraform apply` in ../terraform picks them up
# automatically -- the values never get typed into a terraform command,
# a .tfvars file, or shown anywhere in chat.
#
# Must be SOURCED, not executed, since it needs to modify your current
# shell's environment -- and must be run from inside python/ (the
# ".env" path below is relative to your current directory, not this
# script's location, so this works the same in bash or zsh):
#
#   cd python && source scripts/set_trading_env.sh
#   cd ../terraform && terraform apply
#
# Add DASHBOARD_USERNAME=... and DASHBOARD_PASSWORD=... to python/.env
# yourself first (any username, any real password -- this is what will
# gate the deployed dashboard's Start/Stop/trade buttons) alongside the
# ALPACA_API_KEY_ID/ALPACA_API_SECRET_KEY that should already be there
# from paper-trading setup.

ENV_FILE="$(pwd)/.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "No $ENV_FILE found -- copy python/.env.example to python/.env and fill it in first." >&2
  return 1 2>/dev/null || exit 1
fi

get_var() {
  grep -E "^$1=" "$ENV_FILE" | tail -n1 | cut -d= -f2- | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'\$//"
}

export TF_VAR_alpaca_api_key_id
export TF_VAR_alpaca_api_secret_key
export TF_VAR_dashboard_username
export TF_VAR_dashboard_password
TF_VAR_alpaca_api_key_id="$(get_var ALPACA_API_KEY_ID)"
TF_VAR_alpaca_api_secret_key="$(get_var ALPACA_API_SECRET_KEY)"
TF_VAR_dashboard_username="$(get_var DASHBOARD_USERNAME)"
TF_VAR_dashboard_password="$(get_var DASHBOARD_PASSWORD)"

missing=()
[[ -z "$TF_VAR_alpaca_api_key_id" ]] && missing+=("ALPACA_API_KEY_ID")
[[ -z "$TF_VAR_alpaca_api_secret_key" ]] && missing+=("ALPACA_API_SECRET_KEY")
[[ -z "$TF_VAR_dashboard_username" ]] && missing+=("DASHBOARD_USERNAME")
[[ -z "$TF_VAR_dashboard_password" ]] && missing+=("DASHBOARD_PASSWORD")

if [[ ${#missing[@]} -gt 0 ]]; then
  echo "Missing from $ENV_FILE: ${missing[*]}" >&2
  echo "Add the missing line(s) to python/.env, then re-source this script." >&2
  return 1 2>/dev/null || exit 1
fi

echo "TF_VAR_alpaca_api_key_id, TF_VAR_alpaca_api_secret_key, TF_VAR_dashboard_username, TF_VAR_dashboard_password exported for this shell session (values not printed)."
