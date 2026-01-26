models=("meta-llama/Meta-Llama-3-8B-Instruct" "gpt-4o-mini")
datasets=("humaneval" "cruxeval" "mbpp")

for dataset_name in "${datasets[@]}"; do
    for model_name in "${models[@]}"; do
        echo "Testing model: $model_name on dataset: $dataset_name"
        python eval.py --dataset_name "$dataset_name" --model_name "$model_name" --override_eval
    done
done
