source configs/config.env

# source the env dir, if its not set, error out and ask user to set it with bash runs/create.sh

if [[ -z setup/.venv/bin/activate ]]; then
    echo "Error: env_dir is not set. Please set it with bash runs/create.sh --env_dir <path_to_env_dir>"
    exit 1
fi

source setup/.venv/bin/activate
python configs/create_env_file.py
source configs/config.env

export code_generation_model_save_name="${code_generation_model_name#*/}"
export input_output_prediction_model_save_name="${input_output_prediction_model_name#*/}"

# eval_and_benchmark <save_name> <model_name> [--override_gen] [--override_eval]
# Runs the standard eval+code-benchmark loop for a given set of description predictions.
# save_name:   base name of the description predictions to evaluate
# model_name:  model used for the self code-gen step (pass base model for finetuned pipelines)
# $3:          pass "--override_gen" to re-run baselines.py code steps, or omit/empty to skip
# $4:          pass "--override_eval" to re-run eval.py steps, or omit/empty to skip
eval_and_benchmark() {
    local save_name="$1"
    local model_name="$2"
    local gen_flags=()
    local eval_flags=()
    local dry_run=false
    for arg in "${@:3}"; do
        case "$arg" in
            --override_gen)  gen_flags+=("--override_gen") ;;
            --override_eval) eval_flags+=("--override_eval") ;;
            --echo)          dry_run=true ;;
        esac
    done

    local run
    $dry_run && run="echo" || run=""

    $run python eval.py description --load_name "$save_name" "${eval_flags[@]}"

    local code_eval_name="${save_name}_code_prediction_judge-${code_generation_model_save_name}"
    $run python baselines.py code --load_name "$save_name" --save_name "$code_eval_name" "${gen_flags[@]}"
    $run python eval.py code --load_name "$code_eval_name" "${eval_flags[@]}"

    local code_task_name="${save_name}_code_prediction_judge-self"
    $run python baselines.py code --model_name "$model_name" --load_name "$save_name" --save_name "$code_task_name" "${gen_flags[@]}"
    $run python eval.py code --load_name "$code_task_name" "${eval_flags[@]}"
}