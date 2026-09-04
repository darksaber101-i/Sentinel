#!/usr/bin/env bash
# Loads .env into the environment, then starts Claude Code.
#
# Claude Code expands ${VAR} in .mcp.json from the system environment, but it
# does not read .env files. This launcher bridges that gap so the only thing you
# need on a new machine is a .env file.
#
# Any arguments are forwarded to claude, e.g.  ./start.sh --resume
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$DIR/.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "No .env found at $ENV_FILE"
  echo "Create one with:  cp .env.example .env   (then fill in values)"
  exit 1
fi

# `set -a` marks every subsequent assignment for export, so sourcing .env puts
# the variables into the environment that `claude` inherits.
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

echo "Loaded .env from $ENV_FILE"
exec claude "$@"
