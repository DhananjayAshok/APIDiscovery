source configs/config.env
source $env_dir/bin/activate
python configs/create_env_file.py
source configs/config.env

export code_generation_model_save_name="${code_generation_model_name#*/}"
export input_output_prediction_model_save_name="${input_output_prediction_model_name#*/}"