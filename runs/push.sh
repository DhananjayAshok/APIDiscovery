
exit 1

source scripts/utils.sh

mkdir -p $storage_dir/sync/data/parquets/
mkdir -p $storage_dir/sync/data/finetuning/
mkdir -p $storage_dir/sync/data/csvs/
mkdir -p $storage_dir/sync/data/final/

cp -r $storage_dir/data/parquets/* $storage_dir/sync/data/parquets/
cp -r $storage_dir/data/finetuning/* $storage_dir/sync/data/finetuning/
cp -r $storage_dir/data/csvs/* $storage_dir/sync/data/csvs/
cp -r $storage_dir/data/final/* $storage_dir/sync/data/final/

python sync_data.py push