source configs/config.env | echo "Could not load configs/config.env" && exit 1
if [ -z "$evaluation_model_name" ]; then
    echo "Error: evaluation_model_name is not set. Please set it with python configs/create_env_file.py"
    exit 1
fi
evaluation_model_save_name="${evaluation_model_name#*/}"

models=("meta-llama/Meta-Llama-3-8B-Instruct" "gpt-4o-mini")
datasets=("humaneval" "cruxeval" "mbpp")

for dataset_name in "${datasets[@]}"; do
    for model_name in "${models[@]}"; do
        model_save_name="${model_name#*/}"
        save_name="in_context_$model_save_name"
        echo "Testing: $save_name on dataset: $dataset_name"
        python baselines.py finetuned --dataset_name "$dataset_name" --model_name "$model_name" --save_name "$save_name" # --override_gen
        python eval.py --dataset_name $dataset_name --save_name $save_name # --override_eval,
        #evaluation_output_file=results/$dataset_name/$save_name"_scored_"$evaluation_model_save_name".jsonl"
    done
done
