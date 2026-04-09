source configs/config.env
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
    python eval.py description --save_name $save_name # --override_eval

    #python baselines.py code --model_name $model_name --save_name "$save_name" # --override_gen
    #python eval.py code --save_name $save_name # --override_eval

done