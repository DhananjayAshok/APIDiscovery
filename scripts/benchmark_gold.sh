source configs/config.env
source $env_dir/bin/activate
python configs/create_env_file.py
source configs/config.env
if [ -z "$code_generation_model_name" ]; then
    echo "Error: code_generation_model_name is not set. Please set it with python configs/create_env_file.py"
    exit 1
fi
code_generation_model_save_name="${code_generation_model_name#*/}"
echo "Code generation model: $code_generation_model_save_name"

save_name="code_official"
gold_name="gold_official"
python baselines.py gold_code --save_name "$save_name" # --override_gen
python eval.py code --save_name $gold_name # --override_eval