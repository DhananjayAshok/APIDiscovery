source configs/config.env
if [ -z "$evaluation_model_name" ]; then
    echo "Error: evaluation_model_name is not set. Please set it with python configs/create_env_file.py"
    exit 1
fi
evaluation_model_save_name="${evaluation_model_name#*/}"
echo "Evaluation model: $evaluation_model_save_name"

models=("meta-llama/Meta-Llama-3-8B-Instruct" "Qwen/Qwen3-8B" "Qwen/Qwen3-32B" "gpt-4o-mini")
#datasets=("humaneval" "cruxeval" "mbpp")
#models=("meta-llama/Meta-Llama-3-8B-Instruct")
datasets=("humaneval")


for dataset_name in "${datasets[@]}"; do
    for model_name in "${models[@]}"; do
        model_save_name="${model_name#*/}"
        save_name="zeroshot_$model_save_name"
        echo "Testing: $save_name on dataset: $dataset_name"
        python baselines.py zeroshot --dataset_name "$dataset_name" --model_name "$model_name" --save_name "$save_name" # --override_gen
        python eval.py description --dataset_name $dataset_name --save_name $save_name # --override_eval
        #evaluation_output_file=results/$dataset_name/$save_name"_scored_"$evaluation_model_save_name".jsonl"
        python baselines.py code --dataset_name "$dataset_name" --model_name $model_name --save_name "$save_name" # --override_gen
        python eval.py code --dataset_name $dataset_name --save_name $save_name # --override_eval
        python baselines.py output --dataset_name "$dataset_name" --model_name $model_name --save_name "$save_name" # --override_gen
        python eval.py output --dataset_name $dataset_name --save_name $save_name # --override_eval        
        python baselines.py input --dataset_name "$dataset_name" --model_name $model_name --save_name "$save_name" --override_gen
        python eval.py input --dataset_name $dataset_name --save_name $save_name --override_eval                        
    done
done
