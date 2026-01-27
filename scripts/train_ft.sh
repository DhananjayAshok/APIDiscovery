#!/usr/bin/env bash

source configs/config.sh
# Default values for optional arguments
declare -A ARGS
# make optional arguments: b: batch_size, e: epochs
ARGS["b"]=8   # -b default
ARGS["e"]=5  # -e default
ARGS["d"]="code_alpaca" # dataset_name



# Required arguments
REQUIRED_ARGS=("m")

# Help function
usage() {
    echo "Usage: $0 [-b <value>] [-e <value>] -d <value> -m <value>"
    echo "  -b    Optional (default: ${ARGS["b"]})"
    echo "  -e    Optional (default: ${ARGS["e"]})"
    echo "  -d    Required"
    echo "  -m    Required"
    exit 1
}

# Parse flags
while getopts ":d:b:m:e:" opt; do
    case $opt in
        d|b|m|e)
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
input_file="$storage_dir/data/final/${ARGS["d"]}/train_filtered.jsonl"


bash scripts/llm_utils.sh python train.py --training_kind sft --model_name ${ARGS["m"]} --output_dir $storage_dir/models/$save_name-${ARGS["d"]} \
    --train_file $input_file  --input_column direct_prompt --output_column description --train_validation_split 0.8 \
    --per_device_train_batch_size ${ARGS["b"]} --per_device_eval_batch_size ${ARGS["b"]} \
    --num_train_epochs ${ARGS["e"]} \
    --learning_rate 2e-5 \
    --logging_strategy steps --logging_steps 100 \
    --eval_strategy epoch --eval_steps 0.5 \
    --lora_target_modules k_proj,v_proj,o_proj \
    --save_strategy epoch --save_steps 0.5 \
    --early_stopping_patience 3 \
    --load_best_model_at_end True \
    --run_name ft-$save_name-${ARGS["d"]}