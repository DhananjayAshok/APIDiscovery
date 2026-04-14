source scripts/utils.sh
if [ -z "$code_generation_model_name" ]; then
    echo "Error: code_generation_model_name is not set. Please set it with python configs/create_env_file.py"
    exit 1
fi
echo "Code generation model: $code_generation_model_save_name"
gold_task_name="code_prediction_judge-$code_generation_model_save_name"

echo python baselines.py code --save_name "$gold_task_name" --gold --override_gen
echo python eval.py code --load_name "$gold_task_name" --gold --override_eval