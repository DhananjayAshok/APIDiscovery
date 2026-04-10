source scripts/utils.sh
if [ -z "$evaluation_model_name" ]; then
    echo "Error: evaluation_model_name is not set. Please set it with python configs/create_env_file.py"
    exit 1
fi
evaluation_model_save_name="${evaluation_model_name#*/}"
echo "Evaluation model: $evaluation_model_save_name"

models=("meta-llama/Meta-Llama-3-8B-Instruct" "Qwen/Qwen3-8B" "Qwen/Qwen3-1.7B")
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
    python baselines.py incontext --model_name "$ft_model/final_checkpoint" --save_name "$save_name" #--override_gen
    python eval.py description --load_name $save_name # --override_eval

    # Eval the code with the official model
    code_eval_name="${save_name}_code_prediction_judge-$code_generation_model_save_name"
    python baselines.py code --load_name "$save_name" --save_name $code_eval_name #--override_gen
    python eval.py code --load_name $code_eval_name #--override_eval

    # Write the code with the BASE model (not the ft checkpoint)
    code_task_name="${save_name}_code_prediction_judge-self"
    python baselines.py code --model_name "$model_name" --load_name "$save_name" --save_name $code_task_name #--override_gen
    python eval.py code --load_name $code_task_name #--override_eval

    # Gold: code from true descriptions using the BASE model (not ft checkpoint)
    gold_task_name="$model_save_name"
    #python baselines.py code --model_name "$model_name" --save_name $gold_task_name --gold #--override_gen
    #python eval.py code --load_name $gold_task_name --gold #--override_eval
done
