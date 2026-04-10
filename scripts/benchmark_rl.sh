source scripts/utils.sh
if [ -z "$evaluation_model_name" ]; then
    echo "Error: evaluation_model_name is not set. Please set it with python configs/create_env_file.py"
    exit 1
fi
evaluation_model_save_name="${evaluation_model_name#*/}"
echo "Evaluation model: $evaluation_model_save_name"

models=("Qwen/Qwen3-1.7B")
#bash scripts/warmup_rl.sh -m $model_name

for model_name in "${models[@]}"; do
    model_save_name="${model_name#*/}"
    model_path=$storage_dir/models/rl/$model_save_name/final_checkpoint
    save_name="rl_$model_save_name"
    echo "Testing: $save_name on dataset"
    python baselines.py interactive --model_name $model_path --save_name "$save_name" # --override_gen
    python eval.py description --load_name $save_name # --override_eval

    # Eval the code with the official model
    code_eval_name="${save_name}_code_prediction_judge-$code_generation_model_save_name"
    #python baselines.py code --load_name "$save_name" --save_name $code_eval_name # --override_gen
    #python eval.py code --load_name $code_eval_name # --override_eval

    # Write the code with the BASE model (not the rl checkpoint)
    code_task_name="${save_name}_code_prediction_judge-self"
    #python baselines.py code --model_name "$model_name" --load_name "$save_name" --save_name $code_task_name # --override_gen
    #python eval.py code --load_name $code_task_name # --override_eval

    # Gold: code from true descriptions using the BASE model (not rl checkpoint)
    gold_task_name="$model_save_name"
    #python baselines.py code --model_name "$model_name" --save_name $gold_task_name --gold --override_gen
    #python eval.py code --load_name $gold_task_name --gold --override_eval
done
