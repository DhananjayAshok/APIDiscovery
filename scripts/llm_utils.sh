#!/usr/bin/env bash

# Capture the full command the user wants to run
if [ $# -eq 0 ]; then
    echo "Usage: bash llm_utils.sh <command_to_run>"
    exit 1
fi

# Save the full command (all args)
to_run_command="$@"

currdir="$PWD"

# Enter llm-utils and activate its environment
cd llm-utils || exit 1
source configs/config.env
source "$env_dir/bin/activate" || exit 1

# Return to original directory and load env vars
cd "$currdir" || exit 1
source configs/config.env || exit 1

# Go back into llm-utils where the command should run
cd llm-utils || exit 1

# Run the full command exactly as provided
eval "$to_run_command"

# Return and reactivate project environment
cd "$currdir" || exit 1
source "$env_dir/bin/activate" || exit 1