source configs/config.env
if [ -z "$evaluation_model_name" ]; then
    echo "Error: evaluation_model_name is not set. Please set it with python configs/create_env_file.py"
    exit 1
fi
evaluation_model_save_name="${evaluation_model_name#*/}"
echo "Evaluation model: $evaluation_model_save_name"

#models=("meta-llama/Meta-Llama-3-8B-Instruct" "gpt-4o-mini" "Qwen/Qwen3-8B" "Qwen/Qwen3-32B" "gpt-4o")
#models=("gpt-4o")
#models=("Qwen/Qwen3-8B")
models=("meta-llama/Meta-Llama-3-8B-Instruct" "gpt-4o" "Qwen/Qwen3-8B" "Qwen/Qwen3-32B")


for model_name in "${models[@]}"; do
    model_save_name="${model_name#*/}"
    save_name="incontext_$model_save_name"
    echo "Testing: $save_name"
    python baselines.py incontext --model_name "$model_name" --save_name "$save_name" # --override_gen
    python eval.py description --save_name $save_name # --override_eval,

    #python baselines.py code  --model_name $model_name --save_name "$save_name" # --override_gen
    #python eval.py code  --save_name $save_name # --override_eval

    gold_name="gold_$model_save_name"
    #python baselines.py gold_code  --model_name $model_name --save_name "$save_name" # --override_gen
    #python eval.py code  --save_name $gold_name # --override_eval
done