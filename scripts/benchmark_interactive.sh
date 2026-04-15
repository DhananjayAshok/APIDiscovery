source scripts/utils.sh
if [ -z "$evaluation_model_name" ]; then
    echo "Error: evaluation_model_name is not set. Please set it with python configs/create_env_file.py"
    exit 1
fi
evaluation_model_save_name="${evaluation_model_name#*/}"
echo "Evaluation model: $evaluation_model_save_name"

#models=("meta-llama/Meta-Llama-3-8B-Instruct" "Qwen/Qwen3-32B" "Qwen/Qwen3-8B" "Qwen/Qwen3-1.7B" "Qwen/Qwen3-Coder-30B-A3B-Instruct" "ibm-granite/granite-8b-code-instruct-128k")
#models=("Qwen/Qwen3-Coder-30B-A3B-Instruct" "ibm-granite/granite-8b-code-instruct-128k")
#models=("gpt-4o-mini")
#models=("gpt-4o" "gpt-4o-mini" "gpt-5.4-mini" "Qwen/Qwen3-Coder-30B-A3B-Instruct" "ibm-granite/granite-8b-code-instruct-128k" "Qwen/Qwen3-8B" "Qwen/Qwen3-1.7B" "meta-llama/Meta-Llama-3-8B-Instruct")
#models=("z-ai/glm-5-turbo" "deepseek/deepseek-v3.2") # "google/gemini-3.1-flash-lite-preview")
#models=("Qwen/Qwen3-1.7B")
#models=("meta-llama/Meta-Llama-3-8B-Instruct")
#models=("google/gemini-3.1-pro-preview")
#models=("meta-llama/Meta-Llama-3-8B-Instruct" "Qwen/Qwen3-32B" "Qwen/Qwen3-8B" "Qwen/Qwen3-1.7B" "Qwen/Qwen3-Coder-30B-A3B-Instruct" "ibm-granite/granite-8b-code-instruct-128k")
#models=("z-ai/glm-5-turbo" "deepseek/deepseek-v3.2" "google/gemini-3.1-flash-lite-preview" "gpt-5.4-mini")
models=("anthropic/claude-opus-4.6" "anthropic/claude-sonnet-4.6")

for model_name in "${models[@]}"; do
    model_save_name="${model_name#*/}"
    save_name="interactive_$model_save_name"
    echo "Testing: $save_name "
    echo python baselines.py interactive  --model_name "$model_name" --save_name "$save_name" # --override_gen
    #continue
    echo python eval.py description --load_name $save_name # --override_eval
    
    # Eval the code with the official model
    code_eval_name="$save_name"_code_prediction_judge-$code_generation_model_save_name
    echo python baselines.py code  --load_name "$save_name" --save_name $code_eval_name  # --override_gen
    echo python eval.py code --load_name $code_eval_name # --override_eval
    
    # Write the code with the specific model
    code_task_name="$save_name"_code_prediction_judge-self
    python baselines.py code  --model_name "$model_name" --load_name "$save_name" --save_name $code_task_name  # --override_gen
    python eval.py code --load_name $code_task_name # --override_eval

    # How good is that model at predicting code from gold labels. Set aside for now. 
    gold_task_name="code_prediction_judge-$model_save_name"
    #python baselines.py code --model_name "$model_name" --save_name $gold_task_name --gold --override_gen
    #python eval.py code --load_name $gold_task_name --gold --override_eval
done
