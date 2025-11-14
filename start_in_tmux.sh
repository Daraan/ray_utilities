#!/usr/bin/env bash

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <window-name> [command ...]" >&2
    exit 2
fi

name="$1"
shift

# If no command provided, open an interactive shell with the venv activated.
if [ $# -eq 0 ]; then
    tmux new-window -t session -n "$name" "bash -lc 'source ../env/bin/activate && exec bash'"
    exit 0
fi

# Build a shell-escaped command from the remaining arguments.
cmd=""
for arg in "$@"; do
    printf -v esc '%q ' "$arg"
    cmd+="$esc"
done
cmd=${cmd% }  # remove trailing space

# Create the tmux window and run the command after activating the virtualenv.
tmux new-session -s "$name" -n "$name" -d "bash -c 'source ../env/bin/activate && $cmd; exec bash'"
