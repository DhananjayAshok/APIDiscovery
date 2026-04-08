#!/usr/bin/env bash

declare -A ARGS
ARGS["n"]="4" # Number of devices
ARGS["r"]="" # defaults to the model name if not provided
ARGS["v"]=false # whether to use vllm or APIs for inference. 

# Required arguments
REQUIRED_ARGS=("m")

# Help function
usage() {
    echo "Usage: $0 -m model_name -r run_name [-n num_gpus]"
    echo "Required:"
    echo "  -m model_name     Name of the model to use"
    echo "Options:"
    echo "  -r run_name       Name of the training run (Defaults to model_name if not provided)"
    echo "  -n num_gpus        Number of GPUs to use"
    exit 1
}

# Parse flags
while getopts ":m:r:n:v" opt; do
    case $opt in
        m|r|n|v)
            ARGS["$opt"]="$OPTARG"
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            usage
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            usage
            ;;
    esac
done

# Check required arguments
MISSING=false
for req in "${REQUIRED_ARGS[@]}"; do
    if [[ -z "${ARGS[$req]}" ]]; then
        echo "Error: Missing required argument -$req"
        MISSING=true
    fi
done

if [[ "$MISSING" == true ]]; then
    usage
fi

# Print active variables
echo "Active variables in $0:"
for key in "${!ARGS[@]}"; do
    echo "  -$key = ${ARGS[$key]}"
done

python configs/create_env_file.py
source configs/config.env

# copy the contents of configs/config.env and prepend it to the front of scripts/skyrl/run_rl.sh
{ cat "configs/config.env"; cat scripts/skyrl/run_rl.sh; } > scripts/skyrl/final_run_rl.sh
{ echo "export DATA_DIR=$storage_dir/data/parquets/"; cat scripts/skyrl/final_run_rl.sh; } > temp.txt && mv temp.txt scripts/skyrl/final_run_rl.sh
{  echo "export trainer_policy_model=${ARGS["m"]}"; cat scripts/skyrl/final_run_rl.sh; } > temp.txt && mv temp.txt scripts/skyrl/final_run_rl.sh
{ echo "export run_name=${ARGS["r"]}"; cat scripts/skyrl/final_run_rl.sh; } > temp.txt && mv temp.txt scripts/skyrl/final_run_rl.sh
{ echo "export NUM_GPUS=${ARGS["n"]}"; cat scripts/skyrl/final_run_rl.sh; } > temp.txt && mv temp.txt scripts/skyrl/final_run_rl.sh
{ echo "export USE_VLLM=${ARGS["v"]}"; cat scripts/skyrl/final_run_rl.sh; } > temp.txt && mv temp.txt scripts/skyrl/final_run_rl.sh

mkdir -p SkyRL/skyrl-train/examples/function_discovery/
rm -rf SkyRL/skyrl-train/examples/function_discovery/*
cp -r scripts/skyrl/* SkyRL/skyrl-train/examples/function_discovery/
echo "Copied training scripts to SkyRL/skyrl-train/examples/function_discovery/ Now, you can run the training with the following command:"
echo "bash SkyRL/skyrl-train/examples/function_discovery/final_run_rl.sh"