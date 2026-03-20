source configs/config.env
if [ -z "$evaluation_model_name" ]; then
    echo "Error: evaluation_model_name is not set. Please set it with python configs/create_env_file.py"
    exit 1
fi
evaluation_model_save_name="${evaluation_model_name#*/}"
echo "Evaluation model: $evaluation_model_save_name"

models=("meta-llama/Meta-Llama-3-8B-Instruct" "Qwen/Qwen3-8B" "Qwen/Qwen3-32B")
#models=("Qwen/Qwen3-32B")
#models=("meta-llama/Meta-Llama-3-8B-Instruct")



# First, train each model on the train dataset
for model_name in "${models[@]}"; do
    echo "Training model: $model_name"
    model_save_name="${model_name#*/}"
    bash scripts/train_ft.sh -m $model_name
    # model output dir is $storage_dir/models/$model_save_name-$train_dataset
done

for model_name in "${models[@]}"; do
    model_save_name="${model_name#*/}"
    ft_model="$storage_dir/models/ft/${model_save_name}"
    save_name="ft_${model_save_name}"
    echo "Testing: $save_name" 
    python baselines.py incontext  --model_name "$ft_model/final_checkpoint" --save_name "$save_name" #--override_gen
    python eval.py description   --save_name $save_name # --override_eval,
    #evaluation_output_file=results/$dataset_name/$save_name"_scored_"$evaluation_model_save_name".jsonl"
    python baselines.py code  --model_name $model_name --save_name "$save_name" #--override_gen
    python eval.py code   --save_name $save_name #--override_eval
    python baselines.py output  --model_name $model_name --save_name "$save_name" #--override_gen
    python eval.py output   --save_name $save_name #--override_eval        
    python baselines.py input  --model_name $model_name --save_name "$save_name" #--override_gen
    python eval.py input   --save_name $save_name #--override_eval                

    gold_name="gold_$model_save_name"
    python baselines.py gold_code  --model_name $model_name --save_name "$save_name" #--override_gen
    python eval.py code   --save_name $gold_name #--override_eval
    python baselines.py gold_output  --model_name $model_name --save_name "$save_name" #--override_gen
    python eval.py output   --save_name $gold_name #--override_eval        
    python baselines.py gold_input  --model_name $model_name --save_name "$save_name" #--override_gen
    python eval.py input   --save_name $gold_name #--override_eval                                     

done