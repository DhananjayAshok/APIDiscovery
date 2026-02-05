source configs/config.env | echo "Could not load configs/config.env" && exit 1
if [ -z "$evaluation_model_name" ]; then
    echo "Error: evaluation_model_name is not set. Please set it with python configs/create_env_file.py"
    exit 1
fi
evaluation_model_save_name="${evaluation_model_name#*/}"

models=("meta-llama/Meta-Llama-3-8B-Instruct" "gpt-4o-mini")
datasets=("humaneval" "cruxeval" "mbpp")

train_dataset="code_alpaca" # can change to all later
# First, train each model on the train dataset
for model_name in "${models[@]}"; do
    echo "Training model: $model_name on dataset: $train_dataset"
    model_save_name="${model_name#*/}"
    bash scripts/train_ft.sh -m $model_name -d $train_dataset
    # model output dir is $storage_dir/models/$model_save_name-$train_dataset
done


for dataset_name in "${datasets[@]}"; do
    for model_name in "${models[@]}"; do
        model_save_name="${model_name#*/}"
        ft_model="$storage_dir/models/${model_save_name}-${train_dataset}"
        save_name="ft_${model_save_name}_$train_dataset"
        echo "Testing: $save_name on dataset: $dataset_name"        
        python baselines.py finetuned --dataset_name "$dataset_name" --model_name "$ft_model" --save_name "$save_name" # --override_gen
        python eval.py --dataset_name $dataset_name --save_name $save_name # --override_eval,
        #evaluation_output_file=results/$dataset_name/$save_name"_scored_"$evaluation_model_save_name".jsonl"
    done
done
