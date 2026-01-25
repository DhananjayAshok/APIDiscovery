models=("meta-llama/Meta-Llama-3-8B-Instruct" "Qwen/Qwen3-8B" "mistralai/Mistral-7B-Instruct-v0.3")

for dataset_name in humaneval cruxeval mbpp; do
    for model_name in "${models[@]}"; do
        echo "Testing model: $model_name on dataset: $dataset_name"
        python eval.py --dataset_name "$dataset_name" --model_name "$model_name"
    done
done