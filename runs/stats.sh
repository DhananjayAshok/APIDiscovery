source scripts/utils.sh

python sync_data.py pull
results_dir=$results_dir/


# if results_dir not defined error out
if [ -z "$results_dir" ]; then
    echo "Error: results_dir is not set. Please set it with python configs/create_env_file.py"
    exit 1
fi

mkdir -p $results_dir
cp -r $storage_dir/sync/results/* $results_dir/

python see.py stats
python plot.py

