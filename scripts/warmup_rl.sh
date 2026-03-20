#!/usr/bin/env bash

source configs/config.env
if [[ -z "$storage_dir" ]]; then
    echo "Error: storage_dir is not set. Please set it in configs/config.env."
    exit 1
fi
# Default values for optional arguments
declare -A ARGS
# make optional arguments: b: batch_size, e: epochs
ARGS["b"]=8   # -b default
ARGS["e"]=5  # -e default
ARGS["d"]="gpt-4o-mini" # -d name of the model to use for generating interactive training data, default gpt-4o-mini



# Required arguments
REQUIRED_ARGS=("m")

# Help function
usage() {
    echo "Usage: $0 [-b <value>] [-e <value>] -m <value>"
    echo "  -b    Optional (default: ${ARGS["b"]})"
    echo "  -e    Optional (default: ${ARGS["e"]})"
    echo "  -m    Required"
    exit 1
}

# Parse flags
while getopts ":b:m:e:" opt; do
    case $opt in
        b|m|e)
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
echo "Active variables:"
for key in "${!ARGS[@]}"; do
    echo "  -$key = ${ARGS[$key]}"
done

save_name="${ARGS["m"]#*/}"
generation_model="${ARGS["d"]}"
generation_save_name="${generation_model#*/}"
input_file="$storage_dir/data/finetuning/${generation_save_name}.csv"


bash scripts/llm_utils.sh python train.py --training_kind sft --model_name ${ARGS["m"]} --output_dir $storage_dir/models/rl_warmup/$save_name \
    --train_file $input_file --input_column input --output_column output --train_validation_split 0.85 \
    --per_device_train_batch_size ${ARGS["b"]} --per_device_eval_batch_size ${ARGS["b"]} \
    --num_train_epochs ${ARGS["e"]} \
    --learning_rate 2e-5 \
    --logging_strategy steps --logging_steps 100 \
    --eval_strategy epoch --eval_steps 0.5 \
    --save_strategy epoch --save_steps 0.5 \
    --early_stopping_patience 3 \
    --load_best_model_at_end True \
    --run_name rl_warmup-$save_name
