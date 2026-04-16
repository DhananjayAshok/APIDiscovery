#!/usr/bin/env bash

source scripts/utils.sh || { echo "Could not source utils"; exit 1; }

# Script-specific defaults and required args
declare -A ARGS

REQUIRED_ARGS=("env_dir")

# OPTIONAL: merge shared args from utils.sh (do this BEFORE ALLOWED_FLAGS)
populate_common_optional_training_args ARGS
populate_common_required_training_args REQUIRED_ARGS

# --- Argument parsing (copy verbatim) ---
ALLOWED_FLAGS=("${REQUIRED_ARGS[@]}" "${!ARGS[@]}")
USAGE_STR="Usage: $0"
for req in "${REQUIRED_ARGS[@]}"; do
    USAGE_STR+=" --$req <value>"
done
for opt in "${!ARGS[@]}"; do
    if [[ ! " ${REQUIRED_ARGS[*]} " =~ " ${opt} " ]]; then
        if [[ -z "${ARGS[$opt]}" ]]; then
            echo "DEFAULT VALUE OF KEY \"$opt\" CANNOT BE BLANK"; exit 1
        fi
        USAGE_STR+=" [--$opt <value> (default: ${ARGS[$opt]})]"
    fi
done
function usage() { echo "$USAGE_STR"; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --*)
            FLAG=${1#--}
            VALID=false
            for allowed in "${ALLOWED_FLAGS[@]}"; do
                if [[ "$FLAG" == "$allowed" ]]; then VALID=true; break; fi
            done
            if [ "$VALID" = false ]; then echo "Error: Unknown flag --$FLAG"; usage; fi
            ARGS["$FLAG"]="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

for req in "${REQUIRED_ARGS[@]}"; do
    if [[ -z "${ARGS[$req]}" ]]; then echo "Error: --$req is required."; FAILED=true; fi
done
if [ "$FAILED" = true ]; then usage; fi
# --- End argument parsing ---

# Put your script code below:
root_dir=$(pwd)
#error out if env_dir is not set

if [[ -z "${ARGS["env_dir"]}" ]]; then
    echo "Error: env_dir is not set. Please set it with bash runs/create.sh --env_dir <path_to_env_dir>"
    exit 1
fi

echo "Creating environment in ${ARGS["env_dir"]}. The system will try to save 3 distinct environments in that directory."
# wait for user yes:
read -p "This will create a new environment and install dependencies. Do you want to continue? (y/n) " -n 1 -r
echo    # move to a new line
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborting environment creation."
    exit 1
fi

# set env_dir to 
env_dir="${ARGS["env_dir"]}"/api_env

uv venv "${env_dir}" --python=3.12

cd setup

echo "Simlinking ${env_dir} to .venv for setup scripts"
ln -s "${env_dir}" .venv

echo "Activating environment and installing dependencies"
source .venv/bin/activate

uv sync

echo "API Environment setup complete. You can activate it with 'source setup/.venv/bin/activate' or source scripts/utils.sh"

env_dir="${ARGS["env_dir"]}"/llm-utils_env
keep_going=true
# if llm-utils_env/.venv/bin/activate already exists, skip
if [[ -f "${env_dir}/bin/activate" ]]; then
    echo "llm-utils environment already exists in ${env_dir}, skipping setup."
else
    echo "Setting up llm-utils environment in ${env_dir}"
    uv venv "${env_dir}" --python=3.12
    cd ../llm_utils
    cd ../setup
    echo "Simlinking ${env_dir} to .venv for setup scripts"
    ln -s "${env_dir}" .venv
    echo "Activating environment and installing dependencies"
    source .venv/bin/activate
    uv sync
    # just for this project, manually override the transformers version to avoid SkyRL bugs. 
    uv pip install transformers==4.57.5

    echo "llm-utils Environment setup complete. You Shouldn't have to manually activate it." 
fi

cd $root_dir


skyr_env_dir="${ARGS["env_dir"]}"/skyr_env
if [[ -f "${skyr_env_dir}/bin/activate" ]]; then
    echo "SkyRL environment already exists in ${skyr_env_dir}, skipping setup."
else
    echo "Setting up SkyRL environment in ${skyr_env_dir}"
    uv venv "${skyr_env_dir}" --python=3.12
    ln -s "${skyr_env_dir}" .venv
    source .venv/bin/activate
    uv sync
    uv pip install transformers==4.57.5 # actually .6
fi


cd SkyRL/skyrl-train/

# source the environment for the current session and create the config.env file
echo "Sourcing environment and creating config.env file. If you have [PLACEHOLDER] values in configs/*.yaml, update them."


source scripts/utils.sh

echo "You should see some paths for skyr_env_dir> this should be the same as ${skyr_env_dir}"
