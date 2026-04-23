
#!/usr/bin/env bash

source scripts/utils.sh || { echo "Could not source utils"; exit 1; }

# Script-specific defaults and required args
declare -A ARGS

REQUIRED_ARGS=("model_name")

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


model_name=${ARGS["model_name"]}

model_save_name="${model_name#*/}"

bash scripts/skyrl/move_skyrl.sh -m $model_name -j openrouter -u false

echo "Start a bash job with the command above."
echo "Remember to set you wandb properly." 

# check if the following path exists: $storage_dir/models/rl/$model_save_name/final_checkpoint. If not, error out and ask user to run the training script first.
model_path="$storage_dir/models/rl/$model_save_name/final_checkpoint"
if [[ ! -d "$model_path" ]]; then
    echo "Error: Model checkpoint not found at $model_path. Please run the training script first to proceed."
    exit 1
fi

# tell the user to call on eval:
save_name="rl_$model_save_name"
echo Running: source scripts/utils.sh && python baselines.py interactive  --model_name "$model_path" --save_name "$save_name"
python baselines.py interactive --model_name "$model_path" --save_name "$save_name" --override_gen

echo "Then running: source scripts/utils.sh && eval_and_benchmark --save_name rl_${model_save_name} --model_name $model_name
eval_and_benchmark --save_name "rl_${model_save_name}" --model_name "$model_name"


echo "Running stats pipeline"
python see.py stats