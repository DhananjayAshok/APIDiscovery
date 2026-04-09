source configs/config.env
if [ -z "$evaluation_model_name" ]; then
    echo "Error: evaluation_model_name is not set. Please set it with python configs/create_env_file.py"
    exit 1
fi
evaluation_model_save_name="${evaluation_model_name#*/}"
echo "Evaluation model: $evaluation_model_save_name"

#models=("meta-llama/Meta-Llama-3-8B-Instruct" "Qwen/Qwen3-32B" "Qwen/Qwen3-8B" "Qwen/Qwen3-1.7B")
#models=("Qwen/Qwen3-Coder-30B-A3B-Instruct" "ibm-granite/granite-8b-code-instruct-128k")
#models=("gpt-4o" "gpt-4o-mini" "gpt-5.4-mini")
#models=("gpt-4o" "gpt-4o-mini" "gpt-5.4-mini" "Qwen/Qwen3-Coder-30B-A3B-Instruct" "ibm-granite/granite-8b-code-instruct-128k" "Qwen/Qwen3-8B" "Qwen/Qwen3-1.7B" "meta-llama/Meta-Llama-3-8B-Instruct")
models=("z-ai/glm-5-turbo" "deepseek/deepseek-v3.2") # "google/gemini-3.1-flash-lite-preview")
#models=("Qwen/Qwen3-1.7B")
#models=("meta-llama/Meta-Llama-3-8B-Instruct")
#models=("google/gemini-3.1-pro-preview")


for model_name in "${models[@]}"; do
    model_save_name="${model_name#*/}"
    save_name="interactive_$model_save_name"
    echo "Testing: $save_name "
    python baselines.py interactive  --model_name "$model_name" --save_name "$save_name" # --override_gen
    python eval.py description --save_name $save_name # --override_eval
    
    #python baselines.py code  --model_name $model_name --save_name "$save_name" #--override_gen
    #python eval.py code   --save_name $save_name #--override_eval

    gold_name="gold_$model_save_name"
    #python baselines.py gold_code  --model_name $model_name --save_name "$save_name" # --override_gen
    #python eval.py code   --save_name $gold_name # --override_eval
done