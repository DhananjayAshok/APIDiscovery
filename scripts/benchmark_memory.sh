source scripts/utils.sh
if [ -z "$evaluation_model_name" ]; then
    echo "Error: evaluation_model_name is not set. Please set it with python configs/create_env_file.py"
    exit 1
fi
evaluation_model_save_name="${evaluation_model_name#*/}"
echo "Evaluation model: $evaluation_model_save_name"

models=("gpt-4o-mini")

python retrieval.py embed-train

for model_name in "${models[@]}"; do
    model_save_name="${model_name#*/}"
    save_name="interactive_$model_save_name"
    echo "Testing: $save_name"
    python baselines.py interactive  --model_name "$model_name" --save_name "$save_name" # --override_gen
    python retrieval.py retrieve --load_name $save_name 
    
    new_save_name="memory_$model_save_name"
    echo python baselines.py memory  --model_name "$model_name" --load_name "$save_name" --save_name "$new_save_name" # --override_gen
    python baselines.py memory  --model_name "$model_name" --load_name "$save_name" --save_name "$new_save_name" # --override_gen    
    save_name=$new_save_name
    python eval.py description --load_name $save_name # --override_eval


    
    # Eval the code with the official model
    code_eval_name="$save_name"_code_prediction_judge-$code_generation_model_save_name
    python baselines.py code  --load_name "$save_name" --save_name $code_eval_name  # --override_gen
    python eval.py code --load_name $code_eval_name # --override_eval

    # Write the code with the specific model
    code_task_name="$save_name"_code_prediction_judge-self
    #python baselines.py code  --model_name "$model_name" --load_name "$save_name" --save_name $code_task_name  # --override_gen
    #python eval.py code --load_name $code_task_name # --override_eval

    # How good is that model at predicting code from gold labels. Set aside for now. 
    gold_task_name="code_prediction_judge-$model_save_name"
    #python baselines.py code --model_name "$model_name" --save_name $gold_task_name --gold --override_gen
    #python eval.py code --load_name $gold_task_name --gold --override_eval
done
